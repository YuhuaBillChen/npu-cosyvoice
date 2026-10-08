// libdit_npu: CosyVoice3 DiT (22 layers) with all GEMMs int8 on the XDNA1 NPU (one hw context,
// one 512x1024x1024 i8->i32 kernel) and host ops in C++ (AVX-512, bf16 attention), CFG halves
// pipelined per stage. Variable sequence length; rows are processed in 512-row NPU chunks.
//
// Errors never abort the host process: dit_init / dit_forward* return a negative code and print
// the reason. A priority gate (dit_set_gate) lets another NPU user, e.g. a speech-to-text service,
// pause this library between layers; see npu_cosyvoice/gate.py for the file layout.
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <array>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <immintrin.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ctime>

static const int C = 1024, H = 16, HD = 64, T = 1024, MP = 512, NB = 2;
static int MAXR = 1536, NCH = 3;  // max rows per half, chunks

template <class Tp> static std::vector<Tp> load(const std::string& p, size_t n) {
  std::vector<Tp> v(n); std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("missing " + p);
  f.read((char*)v.data(), n * sizeof(Tp)); return v;
}
static inline uint32_t bf16r(float f) { uint32_t u; memcpy(&u, &f, 4); return (u + 0x7FFF + ((u >> 16) & 1)) >> 16; }
static inline uint32_t pack2(float lo, float hi) { return bf16r(lo) | (bf16r(hi) << 16); }

struct Pool {
  // Workers spin for up to ~2 ms after the last job (keeps per-stage latency low during inference),
  // then sleep on a futex (std::atomic::wait) so an idle library costs no CPU.
  int n; std::vector<std::thread> th; std::atomic<int> gen{0}, next{0}, done{0}; std::atomic<bool> stop{false};
  int count = 0; std::function<void(int)> fn;
  explicit Pool(int n_) : n(n_) {
    for (int i = 1; i < n; i++) th.emplace_back([this] { int seen = 0;
      while (!stop) {
        int g = gen.load(std::memory_order_acquire);
        if (g == seen) {
          auto t0 = std::chrono::steady_clock::now(); bool got = false;
          while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(2)) {
            for (int k = 0; k < 64; k++) _mm_pause();
            if (gen.load(std::memory_order_acquire) != seen || stop) { got = true; break; }
          }
          if (!got) gen.wait(seen, std::memory_order_acquire);
          continue;
        }
        seen = g; work(); done.fetch_add(1);
      } });
  }
  void work() { for (int i; (i = next.fetch_add(1)) < count;) fn(i); }
  void run(int cnt, std::function<void(int)> f) {
    fn = std::move(f); count = cnt; next = 0; done = 0; gen.fetch_add(1, std::memory_order_release); gen.notify_all(); work();
    while (done.load() < n - 1) _mm_pause();
  }
  ~Pool() { stop = true; gen.fetch_add(1); gen.notify_all(); for (auto& t : th) t.join(); }
};

struct Npu {
  xrt::device dev; xrt::hw_context ctx; xrt::kernel k; xrt::bo ibo; uint32_t ninst;
  Npu(const std::string& dir, const std::string& name) : dev(0) {
    xrt::xclbin xb(dir + "/" + name + ".xclbin"); auto uuid = dev.register_xclbin(xb);
    ctx = xrt::hw_context(dev, uuid); k = xrt::kernel(ctx, xb.get_kernels()[0].get_name());
    std::ifstream f(dir + "/" + name + ".bin", std::ios::binary | std::ios::ate); size_t bytes = f.tellg(); ninst = bytes / 4;
    auto ins = load<uint32_t>(dir + "/" + name + ".bin", ninst);
    ibo = xrt::bo(dev, bytes, xrt::bo::flags::cacheable, k.group_id(1)); ibo.write(ins.data()); ibo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  xrt::bo bo(size_t bytes, int g) { return xrt::bo(dev, bytes, xrt::bo::flags::host_only, k.group_id(g)); }
};

struct Gemm { std::vector<xrt::bo> w; std::vector<float> sw, bias; int kb, nb; };
struct Layer { Gemm qkv, o, f1, f2; std::vector<float> inv_smooth; };

static Pool* P; static Npu* NP; static std::vector<Layer> L;

static Gemm make_gemm(const std::string& d, const std::string& nm, int K, int Nn) {
  Gemm g; g.kb = K / T; g.nb = Nn / T;
  for (int a = 0; a < g.kb; a++) for (int b = 0; b < g.nb; b++) {
    auto w = load<int8_t>(d + "/" + nm + "_w_" + std::to_string(a) + "_" + std::to_string(b) + ".bin", (size_t)T * T);
    auto bo = NP->bo((size_t)T * T, 4); bo.write(w.data()); bo.sync(XCL_BO_SYNC_BO_TO_DEVICE); g.w.push_back(bo);
  }
  g.sw = load<float>(d + "/" + nm + "_sw.bin", Nn); g.bias = load<float>(d + "/" + nm + "_b.bin", Nn); return g;
}

struct Half {
  int n = 0, nch = 0;
  std::vector<std::array<xrt::bo, 2>> A; std::vector<std::array<int8_t*, 2>> Am;
  std::vector<std::array<xrt::bo, 3>> Cb; std::vector<std::array<int32_t*, 3>> Cm;
  std::vector<std::vector<xrt::run>> runs; int nrun = 0;  // runs[chunk][slot]
  std::vector<float> xa, xb, x1, att, sx; std::vector<uint32_t> q2, k2, v2;
  float* xin; float* xout; const float* mod;
  Half() {
    A.resize(NCH); Am.resize(NCH); Cb.resize(NCH); Cm.resize(NCH); runs.resize(NCH);
    for (int c = 0; c < NCH; c++) {
      for (int i = 0; i < 2; i++) { A[c][i] = NP->bo((size_t)MP * T, 3); Am[c][i] = A[c][i].map<int8_t*>(); memset(Am[c][i], 0, (size_t)MP * T); }
      for (int i = 0; i < 3; i++) { Cb[c][i] = NP->bo((size_t)MP * T * 4, 5); Cm[c][i] = Cb[c][i].map<int32_t*>(); }
      for (int i = 0; i < 3; i++) runs[c].emplace_back(NP->k);
    }
    xa.resize((size_t)MAXR * C); xb.resize((size_t)MAXR * C); x1.resize((size_t)MAXR * C); att.resize((size_t)MAXR * C); sx.resize(MAXR);
    q2.resize((size_t)MAXR * C / 2); k2.resize((size_t)H * (HD / 2) * MAXR); v2.resize((size_t)H * ((MAXR + 1) / 2) * HD);
  }
  int8_t* arow(int r, int b) { return Am[r / MP][b] + (size_t)(r % MP) * T; }
  int32_t* crow(int r, int c) { return Cm[r / MP][c] + (size_t)(r % MP) * T; }
  void syncA(int nb) { for (int c = 0; c < nch; c++) for (int b = 0; b < nb; b++) A[c][b].sync(XCL_BO_SYNC_BO_TO_DEVICE); }
  void syncC(int nc) { for (int c = 0; c < nch; c++) for (int i = 0; i < nc; i++) Cb[c][i].sync(XCL_BO_SYNC_BO_FROM_DEVICE); }
  // launch slot s on every chunk: A[ch][a] x w -> C[ch][cidx]
  void launch(int s, int a, const xrt::bo& w, int cidx) {
    for (int c = 0; c < nch; c++) { auto& r = runs[c][s]; r.set_arg(0, 3); r.set_arg(1, NP->ibo); r.set_arg(2, NP->ninst); r.set_arg(3, A[c][a]); r.set_arg(4, w); r.set_arg(5, Cb[c][cidx]); r.start(); }
  }
  void wait() { for (int c = 0; c < nch; c++) for (int s = 0; s < nrun; s++) runs[c][s].wait(); }
};
static Half* HS[NB];
static std::vector<float> RC, RS;  // rope cos/sin (n x 64)
static const Layer* CUR;
static int CHUNK = 0;  // >0: block-causal attention (streaming), row i sees keys j < (i/CHUNK+1)*CHUNK

static void quant_rows(Half& hf, const float* src, int K) {
  P->run(hf.n, [&](int r) {
    const float* s = src + (size_t)r * K; float m = 1e-8f;
    for (int c = 0; c < K; c++) m = std::max(m, std::fabs(s[c]));
    float inv = 127.f / m; hf.sx[r] = m / 127.f;
    for (int b = 0; b < K / T; b++) { int8_t* o = hf.arow(r, b); const float* sb = s + b * T; for (int c = 0; c < T; c++) o[c] = (int8_t)std::lrintf(sb[c] * inv); }
  });
}

static void stageA(Half& hf) {  // ln1 + mod + smooth -> quant -> qkv
  const float *sm = hf.mod, *scm = hf.mod + C, *is = CUR->inv_smooth.data();
  P->run(hf.n, [&](int r) {
    const float* xr = hf.xin + (size_t)r * C; double mu = 0, var = 0;
    for (int c = 0; c < C; c++) mu += xr[c]; mu /= C;
    for (int c = 0; c < C; c++) { double d = xr[c] - mu; var += d * d; } var /= C;
    float rs = 1.f / std::sqrt((float)var + 1e-6f), m = (float)mu; float row[C]; float mx = 1e-8f;
    for (int c = 0; c < C; c++) { float v = ((xr[c] - m) * rs * (1.f + scm[c]) + sm[c]) * is[c]; row[c] = v; mx = std::max(mx, std::fabs(v)); }
    float inv = 127.f / mx; hf.sx[r] = mx / 127.f; int8_t* o = hf.arow(r, 0); for (int c = 0; c < C; c++) o[c] = (int8_t)std::lrintf(row[c] * inv);
  });
  hf.syncA(1);
  for (int s = 0; s < 3; s++) hf.launch(s, 0, CUR->qkv.w[s], s); hf.nrun = 3;
}

static void stageB(Half& hf) {  // dequant qkv + rope + bf16 pack, attention, quant -> o
  hf.wait(); hf.syncC(3);
  const int N = hf.n, NH2 = (N + 1) / 2; const Gemm& g = CUR->qkv;
  P->run(N, [&](int r) {
    float s = hf.sx[r]; float qf[C], kf[C];
    const int32_t *cq = hf.crow(r, 0), *ck = hf.crow(r, 1), *cv = hf.crow(r, 2);
    const float *swq = g.sw.data(), *swk = swq + C, *swv = swq + 2 * C, *bq = g.bias.data(), *bk = bq + C, *bv = bq + 2 * C;
    for (int c = 0; c < C; c++) { qf[c] = (float)cq[c] * s * swq[c] + bq[c]; kf[c] = (float)ck[c] * s * swk[c] + bk[c]; }
    const float* cs = RC.data() + (size_t)r * 64; const float* sn = RS.data() + (size_t)r * 64;
    for (float* z : {qf, kf}) { float t0[64]; for (int d = 0; d < 64; d += 2) { t0[d] = z[d] * cs[d] - z[d + 1] * sn[d]; t0[d + 1] = z[d + 1] * cs[d + 1] + z[d] * sn[d + 1]; } memcpy(z, t0, sizeof t0); }
    uint32_t* q2 = hf.q2.data() + (size_t)r * C / 2;
    for (int c = 0; c < C; c += 2) q2[c / 2] = pack2(qf[c] * 0.125f, qf[c + 1] * 0.125f);
    for (int hh = 0; hh < H; hh++) for (int d2 = 0; d2 < HD / 2; d2++) hf.k2[((size_t)hh * (HD / 2) + d2) * N + r] = pack2(kf[hh * HD + 2 * d2], kf[hh * HD + 2 * d2 + 1]);
    uint16_t* v2h = (uint16_t*)hf.v2.data();
    for (int c = 0; c < C; c++) { int hh = c / HD, d = c % HD; v2h[(((size_t)hh * NH2 + r / 2) * HD + d) * 2 + (r & 1)] = (uint16_t)bf16r((float)cv[c] * s * swv[c] + bv[c]); }
  });
  if (N & 1) { uint16_t* v2h = (uint16_t*)hf.v2.data(); for (int hh = 0; hh < H; hh++) for (int d = 0; d < HD; d++) v2h[(((size_t)hh * NH2 + N / 2) * HD + d) * 2 + 1] = 0; }
  const int RB = 8, NP32 = (N + 31) / 32 * 32, nrb = (N + RB - 1) / RB;
  P->run(H * nrb, [&](int task) {
    int hh = task / nrb, r0 = (task % nrb) * RB, nr = std::min(RB, N - r0);
    std::vector<float> Sbuf((size_t)RB * (NP32 + 64)); std::vector<uint32_t> Pbuf((size_t)RB * (NP32 / 2 + 32));
    auto S = [&](int r, int j) -> float& { return Sbuf[(size_t)r * (NP32 + 64) + j]; };
    const uint32_t* K2 = hf.k2.data() + (size_t)hh * (HD / 2) * N;
    uint32_t qp[RB][HD / 2];
    for (int r = 0; r < RB; r++) { int i = r0 + std::min(r, nr - 1); memcpy(qp[r], hf.q2.data() + (size_t)i * C / 2 + hh * HD / 2, sizeof qp[r]); }
    for (int j0 = 0; j0 < N; j0 += 64) {
      int jn = std::min(64, N - j0); __mmask16 mk[4];
      for (int u = 0; u < 4; u++) { int rem = jn - 16 * u; mk[u] = rem >= 16 ? 0xFFFF : (rem <= 0 ? 0 : (__mmask16)((1u << rem) - 1)); }
      __m512 acc[RB][4];
      for (int r = 0; r < RB; r++) for (int u = 0; u < 4; u++) acc[r][u] = _mm512_setzero_ps();
      for (int d2 = 0; d2 < HD / 2; d2++) {
        const uint32_t* kr = K2 + (size_t)d2 * N + j0; __m512bh kv[4];
        for (int u = 0; u < 4; u++) kv[u] = (__m512bh)_mm512_maskz_loadu_epi32(mk[u], kr + 16 * u);
        for (int r = 0; r < RB; r++) { __m512bh qb = (__m512bh)_mm512_set1_epi32((int)qp[r][d2]); for (int u = 0; u < 4; u++) acc[r][u] = _mm512_dpbf16_ps(acc[r][u], qb, kv[u]); }
      }
      for (int r = 0; r < RB; r++) for (int u = 0; u < 4; u++) _mm512_storeu_ps(&S(r, j0 + 16 * u), acc[r][u]);
    }
    const __m512 l2e = _mm512_set1_ps(1.4426950409f), c1 = _mm512_set1_ps(0.6931471806f), c2 = _mm512_set1_ps(0.2402265070f),
                 c3 = _mm512_set1_ps(0.0555041087f), c4 = _mm512_set1_ps(0.0096181291f), c5 = _mm512_set1_ps(0.0013333558f), one = _mm512_set1_ps(1.f);
    for (int r = 0; r < RB; r++) {
      uint32_t* P2 = Pbuf.data() + (size_t)r * (NP32 / 2 + 32);
      if (r >= nr) { memset(P2, 0, sizeof(uint32_t) * (NP32 / 2 + 32)); continue; }
      int lim = CHUNK > 0 ? std::min(N, ((r0 + r) / CHUNK + 1) * CHUNK) : N;
      float mx = -1e30f; for (int j = 0; j < lim; j++) mx = std::max(mx, S(r, j));
      __m512 vmx = _mm512_set1_ps(mx), vsum = _mm512_setzero_ps();
      for (int j = 0; j < NP32; j += 16) {
        int rem = lim - j; __mmask16 m = rem >= 16 ? 0xFFFF : (rem <= 0 ? 0 : (__mmask16)((1u << rem) - 1));
        __m512 xv = _mm512_max_ps(_mm512_mul_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(m, &S(r, j)), vmx), l2e), _mm512_set1_ps(-126.f));
        __m512 fi = _mm512_roundscale_ps(xv, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC); __m512 f = _mm512_sub_ps(xv, fi);
        __m512 p = _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_fmadd_ps(c5, f, c4), f, c3), f, c2), f, c1), f, one);
        __m512 e = _mm512_maskz_mov_ps(m, _mm512_scalef_ps(p, fi)); _mm512_storeu_ps(&S(r, j), e); vsum = _mm512_add_ps(vsum, e);
      }
      float inv = 1.f / _mm512_reduce_add_ps(vsum);
      for (int j = 0; j < NP32; j += 2) P2[j / 2] = pack2(S(r, j) * inv, S(r, j + 1) * inv);
    }
    __m512 o[RB][4];
    for (int r = 0; r < RB; r++) for (int u = 0; u < 4; u++) o[r][u] = _mm512_setzero_ps();
    const uint32_t* V2 = hf.v2.data() + (size_t)hh * NH2 * HD;
    for (int j2 = 0; j2 < NH2; j2++) {
      const uint32_t* vr = V2 + (size_t)j2 * HD; __m512bh vv[4];
      for (int u = 0; u < 4; u++) vv[u] = (__m512bh)_mm512_loadu_si512(vr + 16 * u);
      for (int r = 0; r < RB; r++) { __m512bh pb = (__m512bh)_mm512_set1_epi32((int)Pbuf[(size_t)r * (NP32 / 2 + 32) + j2]); for (int u = 0; u < 4; u++) o[r][u] = _mm512_dpbf16_ps(o[r][u], pb, vv[u]); }
    }
    for (int r = 0; r < nr; r++) for (int u = 0; u < 4; u++) _mm512_storeu_ps(hf.att.data() + (size_t)(r0 + r) * C + hh * HD + 16 * u, o[r][u]);
  });
  quant_rows(hf, hf.att.data(), C); hf.syncA(1);
  hf.launch(0, 0, CUR->o.w[0], 0); hf.nrun = 1;
}

static void stageC(Half& hf) {  // dequant o + residual -> x1, ln2 + mod -> quant -> f1
  hf.wait(); hf.syncC(1);
  const float *gm = hf.mod + 2 * C, *sf = hf.mod + 3 * C, *scf = hf.mod + 4 * C, *sw = CUR->o.sw.data(), *bs = CUR->o.bias.data();
  P->run(hf.n, [&](int r) {
    float s = hf.sx[r]; const int32_t* a = hf.crow(r, 0); const float* xi = hf.xin + (size_t)r * C; float* x1 = hf.x1.data() + (size_t)r * C;
    double mu = 0; for (int c = 0; c < C; c++) { float v = xi[c] + gm[c] * ((float)a[c] * s * sw[c] + bs[c]); x1[c] = v; mu += v; } mu /= C;
    double var = 0; for (int c = 0; c < C; c++) { double d = x1[c] - mu; var += d * d; } var /= C;
    float rs = 1.f / std::sqrt((float)var + 1e-6f), m = (float)mu; float row[C]; float mx = 1e-8f;
    for (int c = 0; c < C; c++) { float v = (x1[c] - m) * rs * (1.f + scf[c]) + sf[c]; row[c] = v; mx = std::max(mx, std::fabs(v)); }
    float inv = 127.f / mx; hf.sx[r] = mx / 127.f; int8_t* o = hf.arow(r, 0); for (int c = 0; c < C; c++) o[c] = (int8_t)std::lrintf(row[c] * inv);
  });
  hf.syncA(1); hf.launch(0, 0, CUR->f1.w[0], 0); hf.launch(1, 0, CUR->f1.w[1], 1); hf.nrun = 2;
}

static void stageD(Half& hf) {  // dequant f1 + gelu -> quant(2048) -> f2
  hf.wait(); hf.syncC(2);
  const float *sw = CUR->f1.sw.data(), *bs = CUR->f1.bias.data();
  P->run(hf.n, [&](int r) {
    float s = hf.sx[r]; float row[2 * C]; float mx = 1e-8f;
    for (int b = 0; b < 2; b++) { const int32_t* a = hf.crow(r, b);
      for (int c = 0; c < T; c++) { float u = (float)a[c] * s * sw[b * T + c] + bs[b * T + c];
        float gg = 0.5f * u * (1.f + std::tanh(0.7978845608f * (u + 0.044715f * u * u * u))); row[b * T + c] = gg; mx = std::max(mx, std::fabs(gg)); } }
    float inv = 127.f / mx; hf.sx[r] = mx / 127.f;
    for (int b = 0; b < 2; b++) { int8_t* o = hf.arow(r, b); for (int c = 0; c < T; c++) o[c] = (int8_t)std::lrintf(row[b * T + c] * inv); }
  });
  hf.syncA(2); hf.launch(0, 0, CUR->f2.w[0], 0); hf.launch(1, 1, CUR->f2.w[1], 1); hf.nrun = 2;
}

static void stageE(Half& hf) {  // dequant f2 (sum K halves) + residual -> xout
  hf.wait(); hf.syncC(2);
  const float *gf = hf.mod + 5 * C, *sw = CUR->f2.sw.data(), *bs = CUR->f2.bias.data();
  P->run(hf.n, [&](int r) {
    float s = hf.sx[r]; const int32_t *a = hf.crow(r, 0), *b2 = hf.crow(r, 1); const float* x1 = hf.x1.data() + (size_t)r * C; float* o = hf.xout + (size_t)r * C;
    for (int c = 0; c < C; c++) o[c] = x1[c] + gf[c] * ((float)(a[c] + b2[c]) * s * sw[c] + bs[c]);
  });
  hf.nrun = 0;
}

// Priority gate: a shared 16-byte file {int32 holders; int32 pad; int64 expiry_ms (CLOCK_REALTIME)}.
// While holders > 0 and the expiry is in the future, forward passes wait at the next layer boundary
// (all NPU runs of the previous layer have completed there). The expiry bounds the pause if a holder
// dies without releasing.
static volatile int32_t* GATE;
static double PAUSED_S;
static int64_t now_ms() { timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000; }
static void gate_wait() {
  if (!GATE) return;
  auto t0 = std::chrono::steady_clock::now(); bool waited = false;
  while (__atomic_load_n(&GATE[0], __ATOMIC_ACQUIRE) > 0 && now_ms() < __atomic_load_n((volatile int64_t*)(GATE + 2), __ATOMIC_ACQUIRE)) {
    waited = true; std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
  if (waited) PAUSED_S += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
extern "C" int dit_set_gate(const char* path) {
  int fd = open(path, O_RDWR | O_CREAT, 0666);
  if (fd < 0) { perror("dit_npu: gate"); return -1; }
  if (ftruncate(fd, 16) != 0) { perror("dit_npu: gate"); close(fd); return -1; }
  void* p = mmap(nullptr, 16, PROT_READ, MAP_SHARED, fd, 0); close(fd);
  if (p == MAP_FAILED) { perror("dit_npu: gate"); return -1; }
  GATE = (volatile int32_t*)p; return 0;
}
// Seconds spent waiting at the gate since init.
extern "C" double dit_paused_seconds() { return PAUSED_S; }

extern "C" int dit_init(const char* aot, const char* wdir, int nlayers, int nthreads, int maxrows) {
  try {
    if (maxrows > 0) { MAXR = maxrows; NCH = (MAXR + MP - 1) / MP; }
    P = new Pool(nthreads); NP = new Npu(aot, "mmi8i32_512x1024x1024_64_64_64");
    char d[1024];
    for (int l = 0; l < nlayers; l++) {
      snprintf(d, sizeof d, "%s/L%02d", wdir, l); std::string s(d); Layer ly;
      ly.qkv = make_gemm(s, "qkv", C, 3 * C); ly.o = make_gemm(s, "o", C, C); ly.f1 = make_gemm(s, "f1", C, 2 * C); ly.f2 = make_gemm(s, "f2", 2 * C, C);
      ly.inv_smooth = load<float>(s + "/qkv_inv_smooth.bin", C); L.push_back(std::move(ly));
    }
    for (int b = 0; b < NB; b++) HS[b] = new Half();
  } catch (const std::exception& e) { fprintf(stderr, "dit_npu: init failed: %s\n", e.what()); return -1; }
  return (int)L.size();
}

// x: [2][n][C] fp32 in/out; mod: [layers][2][6C]; rcos/rsin: [n][64]
extern "C" int dit_forward_chunked(float* x, int n, const float* mod, const float* rcos, const float* rsin, int chunk);
extern "C" int dit_forward(float* x, int n, const float* mod, const float* rcos, const float* rsin) { return dit_forward_chunked(x, n, mod, rcos, rsin, 0); }
extern "C" int dit_forward_chunked(float* x, int n, const float* mod, const float* rcos, const float* rsin, int chunk) {
  if (L.empty() || n > MAXR || n <= 0 || chunk < 0) return -1;
  CHUNK = chunk;
  RC.assign(rcos, rcos + (size_t)n * 64); RS.assign(rsin, rsin + (size_t)n * 64);
  for (int b = 0; b < NB; b++) { HS[b]->n = n; HS[b]->nch = (n + MP - 1) / MP; HS[b]->nrun = 0; memcpy(HS[b]->xa.data(), x + (size_t)b * n * C, sizeof(float) * n * C); }
  void (*stages[5])(Half&) = {stageA, stageB, stageC, stageD, stageE};
  try {
    for (size_t l = 0; l < L.size(); l++) {
      gate_wait();
      CUR = &L[l];
      for (int b = 0; b < NB; b++) { Half& h = *HS[b]; h.mod = mod + ((size_t)l * NB + b) * 6 * C;
        h.xin = (l % 2 == 0) ? h.xa.data() : h.xb.data(); h.xout = (l % 2 == 0) ? h.xb.data() : h.xa.data(); }
      for (auto s : stages) for (int b = 0; b < NB; b++) s(*HS[b]);
    }
  } catch (const std::exception& e) {
    // e.g. DRM_IOCTL_AMDXDNA_EXEC_CMD failing with ENOMEM when the device resumes under memory
    // pressure. Drain whatever is still in flight so the next call starts clean; x is untouched.
    fprintf(stderr, "dit_npu: forward failed: %s\n", e.what());
    for (int b = 0; b < NB; b++) { try { HS[b]->wait(); } catch (...) {} HS[b]->nrun = 0; }
    return -2;
  }
  for (int b = 0; b < NB; b++) { Half& h = *HS[b]; memcpy(x + (size_t)b * n * C, (L.size() % 2 == 0) ? h.xa.data() : h.xb.data(), sizeof(float) * n * C); }
  return 0;
}

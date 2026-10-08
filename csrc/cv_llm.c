// libcv_llm: CosyVoice3's Qwen2 LM backbone through llama.cpp (CPU).
// Minimal C surface over libllama: feed input embeddings, get the final-norm hidden state of the
// last position. CosyVoice keeps its own embeddings, speech-token head and sampling in torch.
#include <string.h>
#include "llama.h"

static struct llama_model * g_model;
static struct llama_context * g_ctx;
static int g_n_embd;
static int g_n_past;

int cv_llm_init(const char * path, int n_ctx, int n_threads) {
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(path, mp);
    if (!g_model) return -1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.no_perf = true;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) return -2;
    g_n_embd = llama_model_n_embd(g_model);
    g_n_past = 0;
    return g_n_embd;
}

void cv_llm_reset(void) {
    llama_memory_clear(llama_get_memory(g_ctx), true);
    g_n_past = 0;
}

// embd: n x n_embd floats appended at the current position; out: n_embd floats
int cv_llm_step(const float * embd, int n, float * out) {
    struct llama_batch b = llama_batch_init(n, g_n_embd, 1);
    memcpy(b.embd, embd, sizeof(float) * (size_t) n * g_n_embd);
    for (int i = 0; i < n; i++) {
        b.pos[i] = g_n_past + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == n - 1;
    }
    b.n_tokens = n;
    int rc = llama_decode(g_ctx, b);
    llama_batch_free(b);
    if (rc != 0) return rc;
    g_n_past += n;
    const float * h = llama_get_embeddings_ith(g_ctx, -1);
    if (!h) return -100;
    memcpy(out, h, sizeof(float) * g_n_embd);
    return 0;
}

// Prefill is compute-bound and decode bandwidth-bound, so callers may switch thread counts per call.
void cv_llm_set_threads(int n) { llama_set_n_threads(g_ctx, n, n); }

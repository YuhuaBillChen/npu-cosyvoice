"""Deadline-driven flow steps for sentence-by-sentence TTS playback.

A client that plays sentence N while sentence N+1 is synthesised needs N+1 before N ends. This picks,
per token2wav call, the most Euler steps (max_steps .. min_steps) whose predicted cost still meets
that deadline, and the cheapest setting for the first sentence of a reply (nothing is playing yet,
so every second is user-visible latency). The cost model is fixed + per (step x frame), calibrated
online from the calls it observes, so it follows whatever else is loading the machine.

Assumes one listener consuming the audio in call order, at real time.
"""
import threading
import time

TOKENS_PER_S = 25  # CosyVoice3 speech tokens per second of audio
MEL_PER_TOKEN = 2


class AdaptiveSteps:
    def __init__(self, min_steps=4, max_steps=10, first_budget_s=0.8, margin_s=0.3,
                 per_step_frame_s=0.0004, fixed_per_audio_s=0.25):
        self.min_steps, self.max_steps = min_steps, max_steps
        self.first_budget_s, self.margin_s = first_budget_s, margin_s
        self.per_step_frame_s = per_step_frame_s    # online estimate (EMA)
        self.fixed_per_audio_s = fixed_per_audio_s  # encoder + vocoder, per second of audio
        self.steps = max_steps
        self.play_end = 0.0
        self.lock = threading.Lock()
        self.last = {}

    def choose(self, frames, audio_s, now):
        first = self.play_end <= now
        budget = self.first_budget_s if first else self.play_end - now - self.margin_s
        fixed = self.fixed_per_audio_s * audio_s
        for st in range(self.max_steps, self.min_steps - 1, -1):
            if fixed + st * frames * self.per_step_frame_s <= budget:
                return st, budget, first
        return self.min_steps, budget, first

    def install(self, model):
        """Wrap a loaded CosyVoice3 model (cosyvoice.cli.cosyvoice.CosyVoice3)."""
        dec = model.model.flow.decoder
        dec_forward = type(dec).forward
        dec.forward = lambda mu, mask, n_timesteps, *a, **k: dec_forward(dec, mu, mask, self.steps, *a, **k)
        token2wav = model.model.token2wav

        def wrapped(*a, **k):
            with self.lock:
                ntok = k["token"].shape[-1]
                frames = k["prompt_feat"].shape[1] + MEL_PER_TOKEN * ntok
                audio_s = ntok / TOKENS_PER_S
                t0 = time.perf_counter()
                self.steps, budget, first = self.choose(frames, audio_s, t0)
                out = token2wav(*a, **k)
                el = time.perf_counter() - t0
                var = max(el - self.fixed_per_audio_s * audio_s, 0.05) / (self.steps * frames)
                self.per_step_frame_s = 0.7 * self.per_step_frame_s + 0.3 * var
                done = time.perf_counter()
                self.play_end = max(done, self.play_end) + audio_s
                self.last = dict(steps=self.steps, first=first, budget_s=round(budget, 2), run_s=round(el, 2), tokens=ntok)
                return out

        model.model.token2wav = wrapped
        return self

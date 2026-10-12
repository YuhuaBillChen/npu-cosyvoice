"""AdaptiveSteps without CosyVoice or an NPU: a fake token2wav reports the steps it was given."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("steps", Path(__file__).resolve().parents[1] / "npu_cosyvoice/steps.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class StepsTests(unittest.TestCase):
    def setUp(self):
        self.steps = m.AdaptiveSteps(per_step_frame_s=0.0001)
        class Decoder:
            def forward(self, mu, mask, n_timesteps):
                return n_timesteps
        dec = Decoder()
        mm = SimpleNamespace(flow=SimpleNamespace(decoder=dec))
        mm.token2wav = lambda **k: self.steps.steps
        self.model = SimpleNamespace(model=mm)
        self.steps.install(self.model)

    def say(self, at, tokens=250):  # 250 tokens = 10 s of audio
        with patch.object(m.time, "perf_counter", return_value=at):
            return self.model.model.token2wav(token=SimpleNamespace(shape=(1, tokens)),
                                              prompt_feat=SimpleNamespace(shape=(1, 50)), speed=1.0)

    def test_reply_starts_cheap_then_ramps_up(self):
        self.assertEqual(self.say(100), 4)    # nothing playing: first sentence
        self.assertEqual(self.say(101), 10)   # 9 s of audio still queued

    def test_new_reply_after_a_pause_starts_cheap(self):
        self.say(100, tokens=625)  # 25 s of audio, estimated to play until 125
        # the user interrupted it; without the reset the 18 s still "queued" would buy 10 steps
        self.assertEqual(self.say(107), 4)

    def test_sentences_within_a_reply_keep_the_buffer(self):
        self.say(100)
        self.assertEqual(self.say(103), 10)


if __name__ == "__main__":
    unittest.main()

"""voice/checksums.py: what setup.sh accepts from a download."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("checksums", Path(__file__).resolve().parents[1] / "voice/checksums.py")
checksums = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checksums)


class ChecksumTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "weights").mkdir()
        (self.dir / "weights/a.bin").write_bytes(b"benign")
        (self.dir / "model.gguf").write_bytes(b"lm")
        self.sums = self.dir / "sums"
        self.sums.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(self.dir)}\n"
                                     for p in (self.dir / "weights/a.bin", self.dir / "model.gguf")))

    def check(self, *exact):
        return checksums.problem(self.sums, self.dir, exact)

    def test_matching_files_pass(self):
        self.assertIsNone(self.check("weights"))

    def test_unlisted_file_in_checked_directory_fails(self):
        # passes `grep | sha256sum -c`, which only reads the listed lines
        (self.dir / "weights/meta.json").write_text("{}")
        self.assertEqual(self.check("weights"), "unexpected file weights/meta.json")

    def test_changed_and_missing_files_fail(self):
        (self.dir / "model.gguf").write_bytes(b"other")
        self.assertEqual(self.check(), "checksum mismatch: model.gguf")
        (self.dir / "model.gguf").unlink()
        self.assertEqual(self.check(), "missing model.gguf")

    def test_published_lists_are_complete(self):
        root = Path(__file__).resolve().parents[1]
        names = [line.split()[1] for line in (root / "voice/cosyvoice3.sha256").read_text().splitlines()]
        for name in ("cosyvoice3.yaml", "llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v3.onnx",
                     "CosyVoice-BlankEN/config.json", "CosyVoice-BlankEN/model.safetensors"):
            self.assertIn(name, names)
        self.assertIn("!voice/cosyvoice3.sha256", (root / ".dockerignore").read_text().split())


if __name__ == "__main__":
    unittest.main()

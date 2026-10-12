import dataclasses
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest.mock import patch

@dataclasses.dataclass
class Options:
    language: str = 'auto'
    prompt: str | None = None
    beam_size: int | None = None

@dataclasses.dataclass
class Result:
    text: str

class DecoderTests(unittest.TestCase):
    def load(self, beam=None):
        self.seen=[]
        def decode(model, mel, options, **kwargs):
            self.seen.append(options)
            return Result('你好,世界!')
        class RawRT:
            def __init__(self, count=6): pass
        modules={'rawxrt':types.SimpleNamespace(RawRT=RawRT),
                 'whisper':types.SimpleNamespace(DecodingOptions=Options,decode=decode),
                 'npu_whisper_encoder':types.SimpleNamespace(prepare_mlp_chunks=lambda *a,**kw:None)}
        env={} if beam is None else {'NPU_STT_BEAM_SIZE':str(beam)}
        runner=runpy.run_path
        with patch.dict(sys.modules,modules),patch.dict(os.environ,env,clear=True),patch('runpy.run_path'),patch('os.chdir'),patch.object(sys,'argv',list(sys.argv)),patch.object(sys,'path',list(sys.path)):
            return runner(str(Path(__file__).resolve().parents[1]/'voice/whisper_entry.py'))['decode']

    def test_default_beam_and_language_detection(self):
        decode=self.load()
        model=types.SimpleNamespace(detect_language=lambda mel:(None,{'zh':.9,'en':.1}))
        out=decode(model,None,Options())
        self.assertIsNone(self.seen[-1].beam_size)  # greedy by default
        self.assertEqual(self.seen[-1].language,'zh')
        self.assertEqual(out.text,'你好，世界！')
        self.assertTrue(self.seen[-1].prompt)

    def test_greedy_opt_out_and_explicit_options(self):
        decode=self.load(1)
        decode(None,None,Options(language='en'))
        self.assertIsNone(self.seen[-1].beam_size)
        self.assertIsNone(self.seen[-1].prompt)
        decode=self.load(5)
        decode(None,None,Options(language='zh',beam_size=3,prompt='custom'))
        self.assertEqual(self.seen[-1].beam_size,3)
        self.assertEqual(self.seen[-1].prompt,'custom')

    def test_invalid_beam_fails_startup(self):
        for beam in (0,9,'bad'):
            with self.subTest(beam=beam),self.assertRaises(ValueError):self.load(beam)

if __name__=='__main__':unittest.main()

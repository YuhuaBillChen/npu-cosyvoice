"""Exercise the proxy without loading model weights or an NPU."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        source=ast.parse((Path(__file__).resolve().parents[1]/'voice/server.py').read_text())
        fn=next(x for x in source.body if isinstance(x,ast.AsyncFunctionDef) and x.name=='transcriptions')
        fn.decorator_list=[];fn.args.args[0].annotation=None
        async def threadpool(call):return call()
        async def body():return b'public audio'
        self.request=SimpleNamespace(body=body,headers={'content-type':'audio/wav'})
        ns={'STT_URL':'http://stt.invalid','urllib':urllib,'json':json,
            'GATE_HOLD':SimpleNamespace(hold=contextlib.nullcontext),
            'run_in_threadpool':threadpool,
            'Response':lambda content,**kwargs:SimpleNamespace(content=content,**kwargs)}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'proxy-under-test','exec'),ns)
        self.call=ns['transcriptions']

    async def test_upstream_error_body_is_not_exposed(self):
        private=b'internal storage path and private upstream credentials'
        for status in (400,413,422,429,500,502,503):
            err=urllib.error.HTTPError('http://stt.invalid',status,'failure',{'Content-Type':'text/html'},io.BytesIO(private))
            with patch('urllib.request.urlopen',side_effect=err):result=await self.call(self.request)
            self.assertEqual(result.status_code,status)
            self.assertNotIn(private,result.content)
            self.assertEqual(result.media_type,'application/json')
            self.assertIn('error',json.loads(result.content))

    async def test_network_error_is_not_exposed(self):
        with patch('urllib.request.urlopen',side_effect=OSError('private connection details')):
            result=await self.call(self.request)
        self.assertEqual(result.status_code,502)
        self.assertNotIn(b'private connection details',result.content)

    async def test_success_body_is_preserved(self):
        body=b'{"text":"hello"}'
        response=SimpleNamespace(status=200,headers={'Content-Type':'application/json'},read=lambda:body)
        with patch('urllib.request.urlopen',return_value=contextlib.nullcontext(response)):
            result=await self.call(self.request)
        self.assertEqual(result.status_code,200)
        self.assertEqual(result.content,body)

if __name__=='__main__':unittest.main()

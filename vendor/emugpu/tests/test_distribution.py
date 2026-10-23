"""Set EMUGPU_WHEEL to verify a built package from an independent directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('EMUGPU_WHEEL')
        if not path: raise unittest.SkipTest('set EMUGPU_WHEEL to test the installed wheel')
        cls.temp=tempfile.TemporaryDirectory(prefix='emugpu-student-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name);cls.site=cls.root/'site';cls.cwd=cls.root/'project';cls.cwd.mkdir()
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(cls.site),str(Path(path).resolve())],
                       capture_output=True,text=True,check=True,timeout=60)
        cls.env=os.environ.copy();cls.env['PYTHONPATH']=str(cls.site);cls.env['CUDA_VISIBLE_DEVICES']=''
        for key in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):cls.env.pop(key,None)
        (cls.cwd/'my_app.py').write_text('''import torch
from emugpu import DataParallel

def worker(ctx):
    mesh=ctx.mesh((ctx.size,),('workers',))
    c=mesh['workers'].comm
    model=DataParallel(torch.nn.Linear(2,1),c)
    model(torch.ones(2,2)).sum().backward()
    model.sync_gradients(2,loss_reduction='sum')
    assert int(c.all_reduce(torch.tensor([1])))==c.size
    return ctx.rank
''')

    def run_python(self,args):
        result=subprocess.run([sys.executable,*args],cwd=self.cwd,env=self.env,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result

    def test_install_excludes_solutions(self):
        self.run_python(['-c', '''import emugpu, importlib.util
assert emugpu.__version__=='0.3.2'
for name in ('emugpu.parallel','emugpu.autograd','emugpu.examples','emugpu.course'):
 assert importlib.util.find_spec(name) is None,name
assert not hasattr(emugpu,'TensorParallelMLP')
'''])

    def test_console_worker_mesh_and_dp(self):
        self.run_python(['-m','emugpu','run','my_app:worker','--world-size','2','--trace','trace.json','--log-dir','logs'])
        trace=json.loads((self.cwd/'trace.json').read_text());self.assertEqual(trace['world_size'],2)
        self.assertTrue((self.cwd/'logs/rank-0.log').is_file())

    def test_resources_and_debug_16(self):
        self.run_python(['-m','emugpu','viewer','--output','viewer.html'])
        self.run_python(['-m','emugpu','debug-config','--world-size','16','--output','debug.json'])
        cfg=json.loads((self.cwd/'debug.json').read_text())
        self.assertEqual(len(cfg['configurations']),16)
        self.assertEqual(cfg['configurations'][-1]['connect']['port'],5693)

if __name__=='__main__':unittest.main()

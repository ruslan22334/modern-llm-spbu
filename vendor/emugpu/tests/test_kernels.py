import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from emugpu import kernels
from emugpu.kernels import cpu_tiles, simulator


class KernelTests(unittest.TestCase):
    def test_masked_strided_vector_and_trace(self):
        x = torch.linspace(-3, 3, 74)[::2]
        y = torch.cos(x)
        trace = []
        torch.testing.assert_close(cpu_tiles.vector_add(x, y, block=16, trace=trace), x + y)
        torch.testing.assert_close(simulator.vector_add(x, y, block=16), x + y)
        self.assertEqual(trace[-1]["valid_lanes"], 5)

    def test_softmax_large_offset_and_stride(self):
        torch.manual_seed(1)
        x = (1000 + torch.randn(3, 74))[:, ::2]
        reference = torch.softmax(x, dim=1)
        torch.testing.assert_close(cpu_tiles.row_softmax(x), reference, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(simulator.row_softmax(x), reference, rtol=2e-6, atol=2e-7)

    def test_tiled_matmul_tail_and_stride(self):
        torch.manual_seed(2)
        a = torch.randn(17, 46)[:, ::2]
        b = torch.randn(19, 23).T
        out = cpu_tiles.tiled_matmul(a, b, bm=16, bn=16, bk=16)
        reference = (a.double() @ b.double()).float()
        torch.testing.assert_close(out, reference, rtol=3e-5, atol=3e-5)

    def test_mask_protects_out_of_bounds(self):
        data = torch.tensor([1.0, 2.0])
        offsets = torch.tensor([0, 1, 100])
        mask = torch.tensor([True, True, False])
        torch.testing.assert_close(simulator.load(data, offsets, mask, other=-1), torch.tensor([1., 2., -1.]))
        with self.assertRaises(IndexError):
            simulator.load(data, offsets)

    def test_import_is_lazy_even_with_gpu_dependencies_installed(self):
        # A fresh interpreter avoids relying on test import order. Any Triton
        # import would raise, proving CPU use has no optional dependency load.
        program = """
import importlib.abc, sys
class BlockTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'triton' or fullname.startswith('triton.'):
            raise AssertionError('CPU imports must not import Triton')
sys.meta_path.insert(0, BlockTriton())
import emugpu.kernels
from emugpu.kernels import cpu_tiles, simulator
assert 'triton' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", program], check=True, cwd=Path(os.path.abspath(os.sep)),
                       timeout=30)

    def test_optional_dependency_error(self):
        if kernels.triton_available():
            self.skipTest("Triton installed: optional dependency error does not apply")
        with self.assertRaisesRegex(RuntimeError, r"emugpu\[triton\]"):
            kernels.add(torch.ones(3), torch.ones(3))


if __name__ == "__main__":
    unittest.main()

import importlib.util
from pathlib import Path
import sys
import unittest

examples=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(examples))
import run_config


class ConfigurationTests(unittest.TestCase):
    def test_catalog_covers_devices_and_mesh(self):
        configs=run_config.configurations()
        for name, config in configs.items():
            report=run_config.describe(config)
            self.assertEqual(len(report['ranks']),report['processes'],name)
            for groups in report['groups'].values():
                self.assertEqual(sorted(r for g in groups for r in g),list(range(report['processes'])),name)
        self.assertEqual(run_config.describe(configs['cpu-32'])['processes'],32)
        hybrid=run_config.describe(configs['cpu-dp4-tpsp2'])
        self.assertEqual(hybrid['groups']['dp'],[[0,2,4,6],[1,3,5,7]])
        self.assertEqual(hybrid['groups']['tpsp'],[[0,1],[2,3],[4,5],[6,7]])
        placement=run_config.describe(configs['gpu8-dp4-two-devices'])
        self.assertEqual(placement['processes'],4)
        self.assertEqual(placement['ranks'][3]['devices'],['cuda:6','cuda:7'])

    def test_two_device_dp_on_cpu(self):
        from emugpu import World
        reports=World(4,devices=[['cpu','cpu'] for _ in range(4)],timeout=120).run(run_config.placement_worker,kwargs={'steps':3})
        self.assertEqual(len({r['pid'] for r in reports}),4)
        self.assertTrue(all(r['status']=='PASS' and r['max_error']<1e-10 for r in reports))

if __name__=='__main__':unittest.main()

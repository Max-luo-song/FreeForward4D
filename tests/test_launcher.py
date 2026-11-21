from pathlib import Path
import importlib.util
import io
import pickle
import tempfile
import unittest
import contextlib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_launcher', ROOT / 'scripts/launch_training.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LaunchRecipes(unittest.TestCase):
    def test_global_batch_is_preserved(self):
        for world in [1, 2, 4, 8]:
            batch, accumulation = launcher.solve_parallel_recipe(64, world, 4)
            self.assertEqual(batch * accumulation * world, 64)
        for world in [1, 2, 4, 8]:
            batch, accumulation = launcher.solve_parallel_recipe(32, world, 2)
            self.assertEqual(batch * accumulation * world, 32)

    def test_invalid_world_batch_combination_rejected(self):
        with self.assertRaises(launcher.LaunchError):
            launcher.solve_parallel_recipe(32, 3, 2)

    def test_public_launch_is_standalone_with_cluster_environment(self):
        output = io.StringIO()
        with patch.dict('os.environ', {'NNODES': '2', 'NODE_RANK': '1',
                                     'MASTER_ADDR': '10.0.0.1'}):
            with contextlib.redirect_stdout(output):
                launcher.main(['--stage', '2s', '--nproc-per-node', '8', '--dry-run'])
        command = output.getvalue().split('command           :')[-1]
        self.assertIn('--standalone', command)
        self.assertNotIn('--nnodes', command)
        self.assertNotIn('--master-addr', command)

    def test_public_launch_rejects_more_than_eight_local_gpus(self):
        with self.assertRaises(launcher.LaunchError):
            launcher.main(['--stage', '2s', '--nproc-per-node', '16', '--dry-run'])

    def test_header_does_not_execute_builtin_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'unexpected'
            expression = f"__import__('pathlib').Path({str(target)!r}).write_text('bad')"
            class Probe:
                def __reduce__(self):
                    return eval, (expression,)
            launcher._HeaderUnpickler(io.BytesIO(pickle.dumps(Probe()))).load()
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()

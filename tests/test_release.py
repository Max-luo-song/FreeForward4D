import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from freeforward.utils.config import merge_config_and_args, load_config
from freeforward.dataset.pose_override import PoseOverrideStore
from freeforward.preprocessing.__main__ import selectors
from freeforward.preprocessing.index import prepare_rgb
from freeforward.preprocessing.pose.data import load_manifest


class ReleaseContracts(unittest.TestCase):
    def test_explicit_default_cli_value_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'config.json'
            p.write_text(json.dumps(dict(batch_size=16, extra_model_field='local')))
            parser = argparse.ArgumentParser()
            parser.add_argument('--batch_size', type=int, default=4)
            args = merge_config_and_args(parser, str(p), ['--batch_size', '4'])
            self.assertEqual(args.batch_size, 4)
            self.assertEqual(args.extra_model_field, 'local')

    def test_unresolved_paths_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'config.json'
            p.write_text('{"data_root":"${FREEFORWARD_TEST_UNSET_3921}/data"}')
            with self.assertRaises(ValueError):
                load_config(p)

    def test_cache_intrinsics_suffix_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key in ['predicted_intrinsics_freeforward', 'predicted_intrinsics_previous_project']:
                path = root / 'scene' / 'omega_pose_override.npz'
                path.parent.mkdir(exist_ok=True)
                np.savez(path, scene_name=np.asarray('scene'), frame_ids=np.array([0]),
                         camera_ids=np.array(['0']), coordinate_frame=np.asarray('global_metric'),
                         omega_camera_to_world_global_metric=np.eye(4)[None],
                         **{key: np.eye(3)[None]})
                value = PoseOverrideStore(root).get_intrinsics('scene', 0, '0')
                np.testing.assert_array_equal(value, np.eye(3))

    def test_gpu_selection_preserves_physical_ids(self):
        self.assertEqual(selectors('2,4', 1), ['2', '4'])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b'}):
            self.assertEqual(selectors(None, 2), ['GPU-a', 'GPU-b'])
        with self.assertRaises(ValueError):
            selectors('0,0', 1)

    def test_rgb_only_manifest_roundtrip_and_resume(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for camera in ['1', '0', '2']:
                folder = root / 'rgb' / 'scene_a' / camera
                folder.mkdir(parents=True)
                for i in range(3):
                    Image.new('RGB', (48, 32), (i * 40, 20, 80)).save(folder / f'{i:06d}.jpg')
            manifest = prepare_rgb(root / 'rgb', root / 'data', 'train')
            before = manifest.read_bytes()
            self.assertEqual(len(load_manifest(manifest)[0]['frames']), 3)
            prepare_rgb(root / 'rgb', root / 'data', 'train')
            self.assertEqual(before, manifest.read_bytes())
            record = json.loads((root / 'data/scene_metadata/train/scene_a.json').read_text())
            self.assertNotIn('camera_to_world', record)
            self.assertNotIn('normalized_intrinsics', record)
            self.assertEqual(record['instances_info'], {})


if __name__ == '__main__':
    unittest.main()

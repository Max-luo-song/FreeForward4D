from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from freeforward.preprocessing.pose.backend.affine_scale import fit
from freeforward.preprocessing.pose.backend.shared_calibration import calibrate_scene


class SharedCalibration(unittest.TestCase):
    def observations(self):
        d = np.linspace(1., 10., 100)
        return [dict(window=w, image=f'image{i}', relative=d / s, metric=d.copy(),
                     metric_reference=5.5) for w, s in enumerate([2., 3.]) for i in [0, 1]]

    def test_exact_metric_scales_and_shared_offsets(self):
        result = fit(2, self.observations())
        np.testing.assert_allclose(result['scales'], [2., 3.], atol=1e-7)
        self.assertTrue(result['converged'])
        self.assertEqual(len(result['offsets_m']), 2)

    def test_unit_rescaling_changes_metric_scales(self):
        observations = self.observations()
        for o in observations:
            o['metric'] *= 100
            o['metric_reference'] *= 100
        result = fit(2, observations)
        np.testing.assert_allclose(result['scales'], [200., 300.], rtol=1e-6)

    def test_scene_calibration_resume_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            d = np.linspace(1., 10., 100)
            for wi, scale in enumerate([2., 3.]):
                start = f'start_{wi:03d}'
                anchors = work / 'scale' / start / 'scene.anchors.npz'
                anchors.parent.mkdir(parents=True)
                raw = np.repeat(np.eye(4)[None], 2, axis=0)
                raw[1, 0, 3] = 1.
                np.savez(anchors, support_band=np.array([scale, scale]),
                         metadata=np.asarray(json.dumps([dict(image='shared', metric_reference=5.5)])),
                         relative_0=d / scale, metric_0=d, raw_c2w=raw,
                         frame_ids=np.array([0, 1]), camera_ids=np.array(['0', '0']))
                output = work / 'windows' / start / 'scene/omega_pose_override.npz'
                output.parent.mkdir(parents=True)
                np.savez(output, omega_c2w_rig_local=raw, rgb_metric_scale=np.asarray(1.))
            scene = dict(scene_name='scene', reference_camera='0', opencv_to_dataset=np.eye(4).tolist())
            calibrate_scene(work, scene, [0, 1])
            path = work / 'windows/start_001/scene/omega_pose_override.npz'
            with np.load(path) as z:
                before = z['omega_c2w_rig_local'].copy()
                self.assertAlmostEqual(before[1, 0, 3], 3.)
            calibrate_scene(work, scene, [0, 1])
            with np.load(path) as z:
                np.testing.assert_array_equal(z['omega_c2w_rig_local'], before)


if __name__ == '__main__':
    unittest.main()

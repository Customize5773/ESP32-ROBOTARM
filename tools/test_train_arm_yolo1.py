"""Offline regression tests for the improved Colab notebook, no robot required.

python -m unittest discover -s tools -p test_train_arm_yolo1.py -v
Uses the same NumPy/OpenCV/ONNX/nbformat/PyYAML dependencies as the notebook.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings
import zipfile

import cv2
import nbformat
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = nbformat.read(ROOT/'Train_arm_yolo1.ipynb', as_version=4)
CELLS = {c.id: c.source for c in NOTEBOOK.cells}


def module(cell):
    result = types.ModuleType(cell)
    exec('\n'.join(CELLS[cell].splitlines()[1:]), result.__dict__)
    return result


RUNTIME = module('runtime-module')
QUALITY = module('quality-module')
MAPPING = module('mapping-module')
DATA = dict(Path=Path, hashlib=hashlib, zipfile=zipfile, yaml=yaml,
            np=np, cv2=cv2, Counter=Counter, warnings=warnings, shutil=shutil)
tree = ast.parse(CELLS['dataset'])
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(tree, 'dataset', 'exec'), DATA)


def object_(cid=0, box=(0, 0, 10, 10), confidence=.9):
    x1, y1, x2, y2 = box
    return dict(class_id=cid, bbox_xyxy=list(box), confidence=confidence,
                center_px=[(x1+x2)/2, (y1+y2)/2],
                contour=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]])


class NotebookTests(unittest.TestCase):
    def test_schema_syntax_and_no_stale_output_or_motion(self):
        nbformat.validate(NOTEBOOK)
        self.assertEqual(len(CELLS), len(NOTEBOOK.cells))
        for cell in NOTEBOOK.cells:
            if cell.cell_type != 'code':
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            source = '\n'.join(l for l in cell.source.splitlines() if not l.startswith('%'))
            ast.parse(source)
            self.assertNotIn('os._exit', source)
            self.assertNotIn('serial.Serial', source)
            self.assertNotIn('V4Serial', source)
        self.assertIn('numpy==2.2.6', CELLS['install'])
        self.assertIn('exist_ok=False', CELLS['bundle'])

    def test_matching_no_duplicate_credit_and_wrong_class(self):
        truth = object_()
        tp, fp, fn = QUALITY.match_objects([object_(), object_(confidence=.8), object_(cid=1)], [truth], .6)
        self.assertEqual((len(tp), len(fp), len(fn)), (1, 2, 0))
        tp, fp, fn = QUALITY.match_objects([object_(cid=1)], [truth], .6)
        self.assertEqual((len(tp), len(fp), len(fn)), (0, 1, 1))
        tp, fp, fn = QUALITY.match_objects([object_(confidence=.5)], [truth], .6)
        self.assertEqual((len(tp), len(fp), len(fn)), (0, 0, 1))

    def test_threshold_selection_rejects_false_positives_and_missing_class(self):
        names = {0:'GREEN', 1:'YELLOW', 2:'RED'}
        records = [dict(image=str(i), shape=(20,20,3), truths=[object_(i)], predictions=[object_(i)]) for i in names]
        records.append(dict(image='negative', shape=(20,20,3), truths=[], predictions=[object_(confidence=.58)]))
        selected = QUALITY.select_threshold(records, names)
        self.assertEqual(selected['selected'], .6)
        self.assertTrue(selected['precision_target_met'])
        summary = QUALITY.summarize(records, names, .6, localization=True)
        self.assertEqual(summary['overall']['tp'], 3)
        self.assertEqual(summary['negative_fp_image_rate'], 0)
        self.assertEqual(summary['matched_centroid_error_px_p95'], 0)
        self.assertEqual(summary['matched_mask_iou_mean'], 1)
        missing = QUALITY.select_threshold(records[:1], names)
        self.assertFalse(missing['precision_target_met'])
        self.assertEqual(missing['selected'], .6)

    def test_no_negatives_is_unknown_not_zero_false_alarm(self):
        record = dict(image='sample', shape=(20,20,3), truths=[object_()], predictions=[])
        result = QUALITY.summarize([record], {0:'GREEN'}, .6, localization=True)
        self.assertIsNone(result['negative_fp_image_rate'])
        self.assertIsNone(result['matched_centroid_error_px_p95'])
        self.assertEqual(result['overall']['fn'], 1)

    def test_mapping_commands_and_workspace(self):
        poses = {'RED':[10,550,125,90,0,0], 'GREEN':[10,410,125,90,0,0], 'YELLOW':[153,530,125,90,0,0]}
        for pose in poses.values():
            self.assertEqual([float(v) for v in MAPPING.command_preview(pose).split()], pose)
            self.assertEqual([float(v) for v in MAPPING.command_preview(pose, 'PARABOLIC', 100).split()], pose+[100])
            self.assertEqual(len(MAPPING.firmware_ik(pose)), 6)
        for pose in [[-1,0,125,90,0,0], [0,0,100,0,0,0], [600,600,800,0,0,0], [10,550,125,90,0,181]]:
            with self.assertRaises(ValueError):
                MAPPING.command_preview(pose)
        with self.assertRaises(ValueError):
            MAPPING.command_preview(poses['RED'], 'PARABOLIC', float('nan'))

    def test_parabolic_checks_entire_path_and_reports_failure(self):
        result = MAPPING.check_parabolic(MAPPING.HOME_POSE, [10,550,125,90,0,0], 100)
        self.assertTrue(result['passed'])
        # Valid endpoints, invalid arc; must fail before any motion can be sent.
        invalid = MAPPING.check_parabolic(MAPPING.HOME_POSE, MAPPING.HOME_POSE, 100)
        self.assertFalse(invalid['passed'])
        self.assertEqual(invalid['waypoint'], 1)

    def test_homography_maps_actual_robot_xy_not_arbitrary_rectangle(self):
        config = dict(frame_size_wh=[640,480], plane_z_mm=125,
                      points_px=[[100,100],[500,100],[500,400],[100,400]],
                      points_robot_xy_mm=[[10,410],[153,410],[153,550],[10,550]])
        cal = MAPPING.RobotPlane(config)
        np.testing.assert_allclose(cal.xy([300,250], (480,640,3)), [81.5,480], atol=.001)
        self.assertIsNone(cal.xy([0,0], (480,640,3)))
        with self.assertRaises(ValueError):
            cal.xy([300,250], (960,1280,3))
        with self.assertRaises(ValueError):
            MAPPING.RobotPlane(dict(config, points_px=[[0,0]]*4))

    def test_runtime_sidecar_hash_confidence_and_actual_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'best.onnx'
            prediction = np.zeros((1, 8, 1), dtype=np.float32)
            prediction[0,:,0] = [16,16,20,20,.8,.1,.1,10]
            proto = np.ones((1,1,8,8), dtype=np.float32)
            nodes = [helper.make_node('Constant', [], ['prediction'], value=numpy_helper.from_array(prediction)),
                     helper.make_node('Constant', [], ['proto'], value=numpy_helper.from_array(proto))]
            graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,32,32])],
                [helper.make_tensor_value_info('prediction',TensorProto.FLOAT,list(prediction.shape)),
                 helper.make_tensor_value_info('proto',TensorProto.FLOAT,list(proto.shape))])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',13)], ir_version=8)
            helper.set_model_props(model, {'names': "{0: 'GREEN', 1: 'YELLOW', 2: 'RED'}"})
            onnx.save(model, path)
            config = dict(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), confidence=.9,
                          preprocessing='stretch_rgb_nchw_fp32')
            sidecar = path.with_name('runtime_config.json')
            sidecar.write_text(json.dumps(config))
            detector = RUNTIME.ColorModel(path)
            frame = np.zeros((32,32,3), np.uint8)
            self.assertEqual(detector.predict(frame), [])
            detected = detector.predict(frame, .6)
            self.assertEqual(len(detected), 1)
            self.assertEqual(detected[0]['color'], 'GREEN')
            config['model_sha256'] = 'wrong'
            sidecar.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'tidak cocok'):
                RUNTIME.ColorModel(path)

    def test_dataset_val_alias_and_decoded_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(42)
            with zipfile.ZipFile(root/'data.zip', 'w') as archive:
                archive.writestr('data.yaml', yaml.safe_dump(dict(names=['HIJAU','KUNING','MERAH'])))
                for split in ('train', 'val', 'test'):
                    for cid in range(3):
                        frame = rng.integers(0,255,(20,20,3),dtype=np.uint8)
                        archive.writestr(f'{split}/images/{split}{cid}.png', cv2.imencode('.png',frame)[1].tobytes())
                        archive.writestr(f'{split}/labels/{split}{cid}.txt', f'{cid} .1 .1 .7 .1 .7 .7 .1 .7')
            base, _, names, _ = DATA['prepare_dataset'](root/'data.zip', root/'work')
            report = DATA['audit_dataset'](base, names)
            self.assertFalse(report['coverage']['all_splits_have_negatives'])
            self.assertEqual(report['valid']['images'], 3)
            # Different encoding but identical decoded pixels, class remains present in valid.
            original = cv2.imread(str(base/'train/images/train0.png'))
            cv2.imwrite(str(base/'valid/images/val0.png'), original, [cv2.IMWRITE_PNG_COMPRESSION,9])
            with self.assertRaisesRegex(ValueError, 'identik lintas split'):
                DATA['audit_dataset'](base, names)

    def test_bundle_contains_matching_reports_mapping_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, evaluation = root/'artifacts', root/'evaluation'
            artifacts.mkdir()
            evaluation.mkdir()
            for filename in ('best.pt', 'best.onnx', 'runtime_config.json'):
                (evaluation/filename).write_text('{}')
            (evaluation/'onnx_quality.json').write_text('{"test": "fixture"}')
            for filename in ('icam_color_runtime.py', 'robot_mapping.py', 'arm_quality.py'):
                (root/filename).write_text('# fixture')
            scope = dict(ARTIFACT_ROOT=artifacts, EVAL_DIR=evaluation, ROOT=root, RUN_ID='test_run',
                         BEST_PT=evaluation/'best.pt', ONNX_PATH=evaluation/'best.onnx',
                         NAMES={0:'GREEN',1:'YELLOW',2:'RED'}, ROBOT_MAPPING={'firmware':'SmoothingPitch.ino'},
                         CALIBRATION={'points_px':None}, calibration=None, REPORT={'coverage':{}},
                         THRESHOLD_REPORT={'precision_target_met':True}, ZIP_SHA256='fixture',
                         ZIP_PATH=root/'dataset.zip', IMGSZ=512, CONF=.6, VALIDATION={}, ONNX_QUALITY={},
                         datetime=datetime, timezone=timezone, shutil=shutil, hashlib=hashlib, json=json,
                         Path=Path, display=lambda *args:None, print=lambda *args:None)
            fake_ipython = types.SimpleNamespace(display=types.SimpleNamespace(FileLink=lambda path:path))
            with patch.dict('sys.modules', {'IPython':fake_ipython}), patch('importlib.metadata.version', return_value='test'):
                exec(CELLS['bundle'], scope)
                bundle = scope['BUNDLE']
                checksums = json.loads((bundle/'SHA256SUMS.json').read_text())
                for filename, digest in checksums.items():
                    self.assertEqual(hashlib.sha256((bundle/filename).read_bytes()).hexdigest(), digest)
                self.assertIn('reports/onnx_quality.json', checksums)
                self.assertIn('robot_mapping.json', checksums)
                self.assertFalse((bundle/'robot_plane.json').exists())
                self.assertTrue(scope['BUNDLE_ZIP'].is_file())
                # A rerun cannot silently mix models and stale camera calibration.
                with self.assertRaises(FileExistsError):
                    exec(CELLS['bundle'], scope)


if __name__ == '__main__':
    unittest.main()

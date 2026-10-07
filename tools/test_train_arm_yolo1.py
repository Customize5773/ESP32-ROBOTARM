"""Offline regression tests for the improved Colab notebook, no robot required.

python -m unittest discover -s tools -p test_train_arm_yolo1.py -v
Uses the same NumPy/OpenCV/ONNX/nbformat/PyYAML dependencies as the notebook.
"""
import ast
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import queue
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
STAGES = module('stage-module')
DATA = dict(Path=Path, hashlib=hashlib, zipfile=zipfile, yaml=yaml,
            np=np, cv2=cv2, Counter=Counter, warnings=warnings, shutil=shutil, json=json)
tree = ast.parse(CELLS['dataset'])
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(tree, 'dataset', 'exec'), DATA)
SOURCE = dict(Path=Path, hashlib=hashlib, zipfile=zipfile, shutil=shutil)
tree = ast.parse(CELLS['dataset-source'])
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
exec(compile(tree, 'dataset-source', 'exec'), SOURCE)


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

    def test_runtime_marks_mat_area_without_removing_green_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root/'scene.png'
            cv2.imwrite(str(image_path), np.zeros((480,640,3), np.uint8))
            config = dict(frame_size_wh=[640,480], plane_z_mm=125,
                          points_px=[[100,100],[500,100],[500,400],[100,400]],
                          points_robot_xy_mm=[[10,410],[153,410],[153,550],[10,550]])
            calibration_path = root/'mat.json'
            calibration_path.write_text(json.dumps(config))
            inside = dict(object_(box=(290,240,310,260)), color='GREEN', center_inside_mask=True)
            outside = dict(object_(box=(0,0,20,20)), color='GREEN', center_inside_mask=True)
            detector = types.SimpleNamespace(predict=lambda *a:[dict(inside),dict(outside)])
            argv = ['runtime', '--source', str(image_path), '--calibration', str(calibration_path)]
            with patch('sys.argv', argv), patch.dict('sys.modules', {'robot_mapping':MAPPING}), \
                 patch.object(RUNTIME, 'ColorModel', return_value=detector), \
                 patch.object(RUNTIME.cv2, 'imwrite', return_value=True), redirect_stdout(io.StringIO()) as output:
                RUNTIME.main()
            detections = json.loads(output.getvalue())['detections']
            self.assertEqual(len(detections),2)
            self.assertEqual([d['color'] for d in detections], ['GREEN','GREEN'])
            self.assertTrue(detections[0]['inside_work_area'])
            self.assertFalse(detections[1]['inside_work_area'])
            self.assertIsNone(detections[1]['xy_mm'])
            np.testing.assert_allclose(detections[0]['xy_mm'],[81.5,480],atol=.001)

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
            for filename in ('icam_color_runtime.py', 'robot_mapping.py', 'arm_quality.py', 'stage_coordinator.py'):
                (root/filename).write_text('# fixture')
            scope = dict(ARTIFACT_ROOT=artifacts, EVAL_DIR=evaluation, ROOT=root, RUN_ID='test_run',
                         BEST_PT=evaluation/'best.pt', ONNX_PATH=evaluation/'best.onnx',
                         NAMES={0:'GREEN',1:'YELLOW',2:'RED'}, ROBOT_MAPPING={'firmware':'SmoothingPitch.ino'},
                         CALIBRATION={'points_px':None}, calibration=None, REPORT={'coverage':{}},
                         STAGE_PROTOCOL={'tcp_adapter_connected':False},
                         THRESHOLD_REPORT={'precision_target_met':True}, DATASET_SHA256='fixture',
                         MERGE_MANIFEST={'sources':[dict(source_id='source_01', zip_name='old.zip',
                             sha256='fixture', id_remap={0:0,1:1,2:2}, attribution={})]},
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




def mini_dataset(path, names, seed, replacements=None):
    rng = np.random.default_rng(seed)
    pixels = {}
    replacements = replacements or {}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data.yaml', yaml.safe_dump(dict(names=names, roboflow={'license':'test'})))
        for split in ('train', 'valid', 'test'):
            for cid in range(3):
                key = (split, cid)
                pixels[key] = replacements.get(key, rng.integers(0,255,(20,20,3),dtype=np.uint8))
                archive.writestr(f'{split}/images/{split}_object{cid}.png', cv2.imencode('.png',pixels[key])[1].tobytes())
                archive.writestr(f'{split}/labels/{split}_object{cid}.txt', f'{cid} .1 .1 .7 .1 .7 .7 .1 .7\n')
    return pixels


class MergeTests(unittest.TestCase):
    def test_remap_preserves_geometry_negative_labels_and_rejects_bbox(self):
        original = '1 .1 .1 .7 .1 .7 .7 .1 .7\n2 .2 .2 .6 .2 .6 .6 .2 .6\n'
        source = {0:'GREEN',1:'RED',2:'YELLOW'}
        canonical = {0:'GREEN',1:'YELLOW',2:'RED'}
        converted = DATA['remap_polygon_labels'](original, source, canonical)
        self.assertEqual([r.split()[0] for r in converted.splitlines()], ['2','1'])
        for before, after in zip(original.splitlines(), converted.splitlines()):
            self.assertEqual(before.split()[1:], after.split()[1:])
        self.assertEqual(DATA['remap_polygon_labels']('\n',source,canonical), '')
        with self.assertRaises(ValueError):
            DATA['remap_polygon_labels']('1 .5 .5 .2 .2',source,canonical)

    def test_merge_remaps_both_sources_keeps_splits_and_same_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root/'old.zip', root/'new.zip'
            mini_dataset(old, ['HIJAU','KUNING','MERAH'], 1)
            mini_dataset(new, ['GREEN','RED','YELLOW'], 2)
            base, yaml_path, names, digest, manifest = DATA['merge_datasets']([old,new],root/'work')
            self.assertEqual(names, {0:'GREEN',1:'YELLOW',2:'RED'})
            self.assertEqual(manifest['merged_images_per_split'], {'train':6,'valid':6,'test':6})
            self.assertEqual(manifest['duplicates_removed'], 0)
            self.assertEqual(manifest['sources'][1]['id_remap'], {0:0,1:2,2:1})
            self.assertEqual(len(manifest['records']), 18)
            self.assertEqual(manifest['sources'][0]['attribution']['license'], 'test')
            label = base/'train/labels/source_02__train_object1.txt'
            self.assertEqual(label.read_text().split()[0], '2')
            self.assertTrue((base/'train/images/source_01__train_object1.png').is_file())
            self.assertTrue((base/'train/images/source_02__train_object1.png').is_file())
            repeat = DATA['merge_datasets']([old,new],root/'work')
            self.assertEqual(repeat[0], base)
            self.assertEqual(repeat[3], digest)
            one = DATA['merge_datasets']([new],root/'work')
            self.assertNotEqual(one[3], digest)
            self.assertEqual(one[2], names)
            self.assertEqual(yaml.safe_load(yaml_path.read_text())['names'], names)

    def test_deduplicate_same_split_after_class_remap_and_keep_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root/'old.zip', root/'new.zip'
            pixels = mini_dataset(old, ['HIJAU','KUNING','MERAH'], 3)
            mini_dataset(new, ['GREEN','RED','YELLOW'], 4, {('train',1):pixels[('train',2)]})
            *_, manifest = DATA['merge_datasets']([old,new],root/'work')
            self.assertEqual(manifest['duplicates_removed'], 1)
            self.assertEqual(manifest['merged_images_per_split']['train'], 5)
            shared = next(r for r in manifest['records'] if len(r['origins']) == 2)
            self.assertEqual({o['source_id'] for o in shared['origins']}, {'source_01','source_02'})
            predictions = [dict(image='some/path/'+r['image']) for r in manifest['records']]
            self.assertEqual(len(DATA['records_for_source'](predictions,manifest['records'],'source_02')),9)

    def test_reject_cross_source_split_leakage_and_conflicting_annotations(self):
        for kind in ('leakage','conflict'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old, new = root/'old.zip', root/'new.zip'
                pixels = mini_dataset(old, ['HIJAU','KUNING','MERAH'], 5)
                replacement = {('test',0):pixels[('train',0)]} if kind == 'leakage' else {('train',0):pixels[('train',2)]}
                mini_dataset(new, ['GREEN','RED','YELLOW'], 6, replacement)
                expected = 'identik lintas split' if kind == 'leakage' else 'Anotasi bertentangan'
                with self.assertRaisesRegex(ValueError, expected):
                    DATA['merge_datasets']([old,new],root/'work')

    def test_resolve_requires_both_zips_and_handles_actual_upload_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mini_dataset(root/'old.zip', ['HIJAU','KUNING','MERAH'], 7)
            with self.assertRaisesRegex(FileNotFoundError, 'new.zip'):
                SOURCE['resolve_dataset_zips'](root,root/'drive',['old.zip','new.zip'])
            def upload(**kwargs):
                mini_dataset(root/'new (1).zip', ['GREEN','RED','YELLOW'], 8)
                return {'new (1).zip':b'uploaded'}
            paths = SOURCE['resolve_dataset_zips'](root,root/'drive',['old.zip','new.zip'],upload)
            self.assertEqual([p.name for p in paths], ['old.zip','new (1).zip'])
            with self.assertRaisesRegex(ValueError, 'dipilih dua kali'):
                SOURCE['resolve_dataset_zips'](root,root/'drive',['old.zip','old.zip'])

    def test_resolve_repairs_escaped_dot_in_bare_zip_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = 'arm_pasti_3_new.v1i.yolov11.zip'
            mini_dataset(root/filename, ['GREEN','RED','YELLOW'], 10)
            paths = SOURCE['resolve_dataset_zips'](
                root, root/'drive', [r'arm_pasti_3_new\.v1i.yolov11.zip'], allow_upload=False)
            self.assertEqual(paths, [(root/filename).resolve()])

    def test_upload_none_automatically_uses_colab_for_missing_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mini_dataset(root/'old.zip', ['HIJAU','KUNING','MERAH'], 11)
            calls = []
            def upload(**kwargs):
                calls.append(kwargs)
                mini_dataset(root/'new (1).zip', ['GREEN','RED','YELLOW'], 12)
                return {'new (1).zip':b'uploaded'}
            google = types.ModuleType('google')
            colab = types.ModuleType('google.colab')
            colab.files = types.SimpleNamespace(upload=upload)
            google.colab = colab
            with patch.dict('sys.modules', {'google':google, 'google.colab':colab}), \
                 patch.dict(SOURCE, {'IN_COLAB':False}):
                paths = SOURCE['resolve_dataset_zips'](root,root/'drive',['old.zip','new.zip'],upload=None)
            self.assertEqual([p.name for p in paths], ['old.zip','new (1).zip'])
            self.assertEqual(calls, [{'target_dir':str(root)}])

    def test_missing_zip_diagnostics_and_explicit_upload_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mini_dataset(root/'available.zip', ['GREEN','RED','YELLOW'], 13)
            def unexpected_upload(**kwargs):
                self.fail('Upload must be disabled')
            with self.assertRaises(FileNotFoundError) as caught:
                SOURCE['resolve_dataset_zips'](root,root/'drive',['missing.zip'],
                                               upload=unexpected_upload, allow_upload=False)
            message = str(caught.exception)
            self.assertIn(str(root/'missing.zip'),message)
            self.assertIn(str(root/'drive/datasets/missing.zip'),message)
            self.assertIn(str(root/'available.zip'),message)




class StageTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.gate = STAGES.StageCoordinator(clock=lambda:self.now, timeout=10.)
        self.detection = dict(color='RED', confidence=.99, xy_mm=[10.,550.],
                              inside_work_area=True, center_inside_mask=True)

    def start(self):
        self.assertTrue(self.gate.on_status({'event':'READY','home':True}))
        self.now += 1
        payload = self.gate.offer([self.detection], self.now)
        self.assertIsNotNone(payload)
        return payload

    def test_initial_home_required_and_lock_precedes_ack(self):
        self.assertIsNone(self.gate.offer([self.detection],self.now))
        self.assertFalse(self.gate.on_status({'event':'READY','home':False}))
        payload = self.start()
        self.assertEqual([payload[k] for k in ('G','R','Y')],[0,1,0])
        self.assertEqual(self.gate.state,'WAIT_ACCEPTED')
        self.assertIsNone(self.gate.offer([dict(self.detection,color='GREEN')],self.now))
        self.assertFalse(self.gate.on_status('OK'))
        self.assertEqual(self.gate.state,'WAIT_ACCEPTED')
        payload['x'] = -123
        self.detection['xy_mm'][0] = -456
        self.assertEqual(self.gate.active['x'],10.)

    def test_stage_finishes_only_at_home_then_new_frame_no_queue(self):
        first = self.start()
        self.assertTrue(self.gate.on_status({'event':'ACCEPTED','stage_id':first['stage_id']}))
        self.now += 1
        pending_frame = self.now
        another = dict(self.detection,color='GREEN')
        self.assertIsNone(self.gate.offer([another],pending_frame))
        self.assertFalse(self.gate.on_status({'event':'READY','home':True}))
        self.assertFalse(self.gate.on_status({'event':'DONE','stage_id':'another_stage','home':True}))
        self.assertEqual(self.gate.state,'BUSY')
        self.now += 1
        self.assertTrue(self.gate.on_status({'event':'DONE','stage_id':first['stage_id'],'home':True}))
        self.assertIsNone(self.gate.active)
        self.assertIsNone(self.gate.offer([another],pending_frame))
        self.assertIsNone(self.gate.offer([another],self.now))
        self.now += .1
        second = self.gate.offer([another],self.now)
        self.assertNotEqual(second['stage_id'],first['stage_id'])
        self.assertEqual([second[k] for k in ('G','R','Y')],[1,0,0])

    def test_done_without_home_or_acceptance_faults(self):
        for accepted, home in [(True,False),(False,True)]:
            self.setUp()
            first = self.start()
            if accepted:
                self.gate.on_status({'event':'ACCEPTED','stage_id':first['stage_id']})
            self.assertFalse(self.gate.on_status({'event':'DONE','stage_id':first['stage_id'],'home':home}))
            self.assertEqual(self.gate.state,'FAULT')
            self.assertIsNone(self.gate.offer([self.detection],self.now))

    def test_timeout_reset_error_and_disconnect_never_retry(self):
        for failure in ('timeout','reset','error','disconnect'):
            self.setUp()
            first = self.start()
            if failure == 'timeout':
                self.now += 10
                self.gate.tick()
            elif failure == 'reset':
                self.gate.on_status({'event':'RESET'})
            elif failure == 'error':
                self.gate.on_status({'event':'ERROR','stage_id':first['stage_id'],'reason':'grip failed'})
            else:
                self.gate.fail('TCP connection lost')
            self.assertEqual(self.gate.state,'FAULT')
            self.assertFalse(self.gate.on_status({'event':'READY','home':True}))
            self.assertFalse(self.gate.on_status({'event':'DONE','stage_id':first['stage_id'],'home':True}))
            self.assertIsNone(self.gate.offer([self.detection],self.now))
            self.assertEqual(self.gate.active,first)

    def test_selection_requires_fresh_frame_valid_mat_and_color(self):
        self.gate.on_status({'event':'READY','home':True})
        self.now += 2
        for stamp in (100.,100.1,103.,float('nan')):
            self.assertIsNone(self.gate.offer([self.detection],stamp))
        bad = [
            dict(self.detection,inside_work_area=False),
            dict(self.detection,center_inside_mask=False),
            dict(self.detection,color='MAT'),
            dict(self.detection,confidence=.1),
            dict(self.detection,xy_mm=[float('inf'),10.]),
        ]
        self.assertIsNone(self.gate.offer(bad,self.now))
        good = dict(self.detection,color='YELLOW',confidence=.9)
        winner = self.gate.offer([good,self.detection],self.now)
        self.assertEqual(winner['R'],1)

    def test_camera_rejects_queued_frames_from_before_done(self):
        camera = RUNTIME.CamNaviSource.__new__(RUNTIME.CamNaviSource)
        camera.frames = queue.Queue()
        camera.frames.put((99.8,np.zeros((2,2,3),np.uint8)))
        camera.frames.put((99.95,np.ones((2,2,3),np.uint8)))
        with patch.object(RUNTIME.time,'monotonic',return_value=100.):
            stamp, frame = camera.read_sample(after=99.9)
        self.assertEqual(stamp,99.95)
        self.assertTrue(frame.all())

    def test_camera_timestamp_is_not_refreshed_by_slow_decode(self):
        camera = RUNTIME.CamNaviSource.__new__(RUNTIME.CamNaviSource)
        camera.frames = queue.Queue(maxsize=1)
        now = [99.8]
        buffer = types.SimpleNamespace(get_size=lambda:1, extract_dup=lambda *a:b'0')
        sample = types.SimpleNamespace(get_buffer=lambda:buffer)
        def decode(*args):
            now[0] = 100.2  # DONE could arrive while JPEG decoding is still in progress.
            return np.ones((2,2,3),np.uint8)
        with patch.object(RUNTIME.time,'monotonic',side_effect=lambda:now[0]), \
             patch.object(RUNTIME.cv2,'imdecode',side_effect=decode):
            camera._sample(sample)
        self.assertEqual(camera.frames.get_nowait()[0],99.8)


if __name__ == '__main__':
    unittest.main()

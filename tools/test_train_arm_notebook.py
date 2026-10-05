"""Offline checks for the self-contained training notebook (no camera/robot needed).

python -m unittest discover -s tools -p test_train_arm_notebook.py -v
Requires numpy, opencv-python(-headless), PyYAML, nbformat, onnx, onnxruntime.
"""
import ast
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import shutil
import tempfile
import time
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
NOTEBOOK = nbformat.read(ROOT / 'Train-arm-yolo.ipynb', as_version=4)
CELLS = {c.id: c.source for c in NOTEBOOK.cells}
RUNTIME = types.ModuleType('notebook_runtime')
exec('\n'.join(CELLS['runtime-module'].splitlines()[1:]), RUNTIME.__dict__)
DATA = dict(Path=Path, hashlib=hashlib, zipfile=zipfile, yaml=yaml,
            np=np, cv2=cv2, Counter=Counter, warnings=warnings)
definitions = ast.parse(CELLS['dataset'])
definitions.body = [node for node in definitions.body if isinstance(node, ast.FunctionDef)]
exec(compile(definitions, 'dataset-cell', 'exec'), DATA)
SOURCE = dict(Path=Path, hashlib=hashlib, shutil=shutil, zipfile=zipfile)
source_definitions = ast.parse(CELLS['dataset-source'])
source_definitions.body = [node for node in source_definitions.body if isinstance(node, ast.FunctionDef)]
exec(compile(source_definitions, 'dataset-source-cell', 'exec'), SOURCE)


class FakeSerial:
    def __init__(self, reply):
        self.reply, self.lines, self.writes, self.closed = reply, [], [], False

    def reset_input_buffer(self):
        self.lines.clear()

    def write(self, command):
        self.writes.append(command)
        if command == b'POINTS\n':
            self.lines = [f'Titik {i} (0,0) : 0 | 0 | 0 | 0 | 0 deg' for i in range(1, 5)]
            self.lines.append('Area W x H = 300.00 x 200.00 mm')
        elif command.startswith(b'XY:'):
            self.lines = list(self.reply)

    def readline(self):
        return (self.lines.pop(0)+'\n').encode() if self.lines else b''

    def close(self):
        self.closed = True


class NotebookTests(unittest.TestCase):
    def test_colab_a100_configuration_and_gpu_required(self):
        config = ast.parse(CELLS['config'])
        config.body = [node for node in config.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # GPU configuration should work before a dataset has been uploaded.
            for gpu, vram, expected in [('NVIDIA A100-SXM4-40GB', 40, 64), ('Tesla T4', 16, 16), ('CPU', 0, None)]:
                cuda = types.SimpleNamespace(is_available=lambda: vram > 0,
                    get_device_name=lambda n: gpu,
                    get_device_properties=lambda n: types.SimpleNamespace(total_memory=vram*1024**3))
                scope = dict(Path=Path, ROOT=root, PERSIST=root, IN_COLAB=True, shutil=shutil,
                             os=os, torch=types.SimpleNamespace(cuda=cuda))
                with redirect_stdout(io.StringIO()):
                    if expected is None:
                        with self.assertRaisesRegex(RuntimeError, 'GPU belum aktif'):
                            exec(compile(config, 'config-cell', 'exec'), scope)
                    else:
                        exec(compile(config, 'config-cell', 'exec'), scope)
                        self.assertEqual(scope['BATCH'], expected)
                        self.assertEqual(scope['DEVICE'], 0)
                        self.assertLessEqual(scope['WORKERS'], 4)

    def test_schema_and_python_cells(self):
        nbformat.validate(NOTEBOOK)
        for cell in NOTEBOOK.cells:
            if cell.cell_type != 'code':
                continue
            self.assertIsNone(cell.execution_count)
            self.assertEqual(cell.outputs, [])
            lines = cell.source.splitlines()
            if lines[0].startswith('%%writefile'):
                lines = lines[1:]
            ast.parse('\n'.join(line for line in lines if not line.startswith('%')), filename=cell.id)

    def test_motion_disabled_by_default(self):
        assignments = {n.targets[0].id: ast.literal_eval(n.value)
                       for n in ast.parse(CELLS['move']).body if isinstance(n, ast.Assign)}
        self.assertIs(assignments['SEND_ONE_TARGET'], False)
        self.assertIs(assignments['CALIBRATION_VERIFIED'], False)

    def test_actual_zip_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            base, data_yaml, names, digest = DATA['prepare_dataset'](
                ROOT/'arm-warna-3.v5i.yolov11.zip', Path(directory))
            with warnings.catch_warnings(record=True) as caught:
                report = DATA['audit_dataset'](base, names)
            self.assertEqual(names, {0: 'GREEN', 1: 'YELLOW', 2: 'RED'})
            self.assertEqual([report[s]['images'] for s in ('train', 'valid', 'test')], [3480, 233, 94])
            self.assertEqual(report['train']['per_class'], {'RED': 924, 'GREEN': 1124, 'YELLOW': 1432})
            self.assertEqual(report['train']['large_bbox_fraction'], 1.0)
            self.assertTrue(any('crop besar' in str(w.message) for w in caught))
            self.assertEqual(yaml.safe_load(data_yaml.read_text())['path'], str(base.resolve()))
            self.assertEqual(len(digest), 64)

    def test_zip_rejects_path_escape_and_conflicting_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for entries in [[('../escape.txt', 'x')], [('data.yaml', 'a'), ('data.yaml', 'b')]]:
                archive = root/'bad.zip'
                with zipfile.ZipFile(archive, 'w') as z, warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    for name, content in entries:
                        z.writestr(name, content)
                with self.assertRaises(ValueError):
                    DATA['prepare_dataset'](archive, root/'work')
            self.assertFalse((root/'escape.txt').exists())


class DatasetSourceTests(unittest.TestCase):
    def make_zip(self, path, content='names: [GREEN, YELLOW, RED]'):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('data.yaml', content)
        return path

    def test_local_zip_does_not_open_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.make_zip(root/'arm-warna-3.v5i.yolov11.zip')
            upload = unittest.mock.Mock(side_effect=AssertionError('Upload should not open'))
            self.assertEqual(SOURCE['resolve_dataset_zip'](root, root/'drive', upload=upload), path.resolve())
            upload.assert_not_called()

    def test_drive_copy_and_explicit_source_preserve_originals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            persist = root/'drive'
            source = self.make_zip(persist/'datasets/arm-warna-3.v5i.yolov11.zip')
            result = SOURCE['resolve_dataset_zip'](root, persist)
            self.assertNotEqual(result, source.resolve())
            self.assertEqual(result.read_bytes(), source.read_bytes())
            original = self.make_zip(root/'arm-warna-3.v5i.yolov11.zip', 'old data')
            before = original.read_bytes()
            chosen = self.make_zip(persist/'folder with spaces/custom (1).zip', 'new data')
            result = SOURCE['resolve_dataset_zip'](root, persist, source=str(chosen))
            self.assertEqual(result.read_bytes(), chosen.read_bytes())
            self.assertEqual(original.read_bytes(), before)

    def test_upload_uses_saved_name_with_duplicate_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def upload(target_dir):
                calls.append(target_dir)
                path = self.make_zip(Path(target_dir)/'arm-warna-3.v5i.yolov11 (1).zip')
                return {str(path): path.read_bytes()}
            with redirect_stdout(io.StringIO()):
                result = SOURCE['resolve_dataset_zip'](root, root/'drive', upload=upload)
            self.assertEqual(calls, [str(root)])
            self.assertTrue(result.is_file())
            self.assertTrue(result.name.endswith(' (1).zip'))

    def test_missing_cancelled_multiple_and_invalid_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                SOURCE['resolve_dataset_zip'](root, root/'drive')
            upload = unittest.mock.Mock()
            with self.assertRaises(FileNotFoundError):
                SOURCE['resolve_dataset_zip'](root, root/'drive', source='missing.zip', upload=upload)
            upload.assert_not_called()
            for response in ({}, {'a.zip': b'a', 'b.zip': b'b'}):
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    SOURCE['resolve_dataset_zip'](root, root/'drive', upload=lambda **kw: response)
            bad = root/'bad.zip'
            bad.write_bytes(b'not an archive')
            with self.assertRaisesRegex(ValueError, 'bukan arsip ZIP'):
                SOURCE['resolve_dataset_zip'](root, root/'drive', source=bad)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.names = {0: 'GREEN', 1: 'YELLOW', 2: 'RED'}
        self.config = dict(frame_size_wh=[640, 480], area_mm=[300, 200],
                           corners_px=[[50, 430], [590, 430], [590, 50], [50, 50]])

    def raw_outputs(self):
        # Two duplicate GREEN boxes and overlapping RED: class-aware NMS keeps two.
        rows = np.array([[256, 256, 100, 100, .9, .01, .01, 1],
                         [256, 256, 100, 100, .8, .01, .01, 1],
                         [256, 256, 100, 100, .01, .01, .95, 1]], dtype=np.float32)
        return rows.T[None], np.ones((1, 1, 128, 128), np.float32)

    def test_segmentation_decode_nms_and_non_square_coordinates(self):
        result = RUNTIME.decode(*self.raw_outputs(), (480, 640, 3), (512, 512), self.names)
        self.assertEqual([r['color'] for r in result], ['RED', 'GREEN'])
        np.testing.assert_allclose(result[0]['center_px'], [320, 240], atol=2)
        self.assertTrue(result[0]['center_inside_mask'])
        self.assertEqual(RUNTIME.decode(*self.raw_outputs(), (480, 640, 3), (512, 512), self.names, conf=.99), [])

    def test_invalid_output_and_empty_masks(self):
        pred, proto = self.raw_outputs()
        with self.assertRaises(ValueError):
            RUNTIME.decode(pred.transpose(0, 2, 1), proto, (480, 640, 3), (512, 512), self.names)
        self.assertEqual(RUNTIME.decode(pred, -proto, (480, 640, 3), (512, 512), self.names), [])
        pred[:] = np.nan
        self.assertEqual(RUNTIME.decode(pred, proto, (480, 640, 3), (512, 512), self.names), [])

    def test_onnx_runtime_session_and_metadata(self):
        pred, proto = self.raw_outputs()
        nodes = [helper.make_node('Constant', [], ['prediction'], value=numpy_helper.from_array(pred)),
                 helper.make_node('Constant', [], ['prototype'], value=numpy_helper.from_array(proto))]
        graph = helper.make_graph(nodes, 'test-segment',
            [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 512, 512])],
            [helper.make_tensor_value_info('prediction', TensorProto.FLOAT, list(pred.shape)),
             helper.make_tensor_value_info('prototype', TensorProto.FLOAT, list(proto.shape))])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)], ir_version=9)
        helper.set_model_props(model, {'names': repr(self.names)})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.onnx'
            onnx.save(model, path)
            runtime = RUNTIME.ColorModel(path)
            frame = np.zeros((480, 640, 3), np.uint8)
            self.assertEqual(len(runtime.predict(frame)), 2)
            with self.assertRaises(ValueError):
                runtime.predict(frame[:, :, 0])
            helper.set_model_props(model, {'names': "{0: 'HIJAU', 1: 'KUNING', 2: 'MERAH'}"})
            onnx.save(model, path)
            with self.assertRaises(ValueError):
                RUNTIME.ColorModel(path)

    def test_calibration_corners_center_and_outside(self):
        cal = RUNTIME.PlaneCalibration(self.config)
        for source, expected in zip(self.config['corners_px'], [[0, 0], [300, 0], [300, 200], [0, 200]]):
            np.testing.assert_allclose(cal.xy(source, (480, 640, 3)), expected, atol=.001)
        np.testing.assert_allclose(cal.xy([320, 240], (480, 640, 3)), [150, 100], atol=.001)
        self.assertIsNone(cal.xy([10, 10], (480, 640, 3)))
        with self.assertRaises(ValueError):
            cal.xy([320, 240], (960, 1280, 3))
        with self.assertRaises(ValueError):
            cal.xy([float('nan'), 2], (480, 640, 3))

    def test_calibration_rejects_missing_crossed_degenerate(self):
        for points in [None, [[0, 0]]*4, [[0, 0], [600, 400], [600, 0], [0, 400]]]:
            with self.assertRaises(ValueError):
                RUNTIME.PlaneCalibration(dict(self.config, corners_px=points))

    def test_dry_run_and_invalid_target(self):
        robot = RUNTIME.V4Serial(dry_run=True)
        self.assertIsNone(robot.serial)
        self.assertEqual(robot.move_xy(150, 100), 'XY:150.00,100.00')
        for x, y in [(301, 100), (-1, 0), (float('nan'), 0), (0, float('inf'))]:
            with self.assertRaises(ValueError):
                robot.move_xy(x, y)

    def serial_robot(self, reply, **kwargs):
        fake = FakeSerial(reply)
        stub = types.SimpleNamespace(Serial=lambda *a, **kw: fake)
        with patch.dict('sys.modules', {'serial': stub}), patch.object(RUNTIME.time, 'sleep'):
            robot = RUNTIME.V4Serial('FAKE', dry_run=False, **kwargs)
        return robot, fake

    def test_serial_waits_for_completion_or_already_there(self):
        for terminal in ['Motion selesai', 'Sudah di posisi target']:
            robot, fake = self.serial_robot(['XY -> (150, 100)', '  axis0 -> 10 deg', terminal])
            self.assertEqual(robot.move_xy(150, 100), 'XY:150.00,100.00')
            self.assertEqual(fake.writes, [b'POINTS\n', b'XY:150.00,100.00\n'])
            robot.close()
            self.assertTrue(fake.closed)

    def test_serial_rejects_errors_and_latches_fault(self):
        for reply in ['LIMIT axis 0 : 190', 'ZERO belum dibuat', 'Kalibrasi belum lengkap', 'STOP', ' ESP32 ROBOT ARM - KALIBRASI 4 TITIK']:
            robot, fake = self.serial_robot(['XY -> (150, 100)', reply])
            with self.assertRaises(RuntimeError):
                robot.move_xy(150, 100)
            self.assertEqual(fake.writes[-1], b'STOP\n')
            count = len(fake.writes)
            with self.assertRaises(RuntimeError):
                robot.move_xy(150, 100)
            self.assertEqual(len(fake.writes), count)

    def test_serial_timeout_stops_and_area_mismatch_closes(self):
        robot, fake = self.serial_robot(['XY -> (150, 100)'], timeout=.001)
        with self.assertRaises(TimeoutError):
            robot.move_xy(150, 100)
        self.assertEqual(fake.writes[-1], b'STOP\n')
        fake = FakeSerial([])
        with patch.dict('sys.modules', {'serial': types.SimpleNamespace(Serial=lambda *a, **kw: fake)}), patch.object(RUNTIME.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                RUNTIME.V4Serial('FAKE', area_mm=(400, 300), dry_run=False)
        self.assertTrue(fake.closed)

    def test_camera_rejects_stale_frames(self):
        camera = RUNTIME.CamNaviSource.__new__(RUNTIME.CamNaviSource)
        camera.frames = queue.Queue(maxsize=1)
        camera.frames.put((time.monotonic()-10, np.zeros((10, 10, 3), np.uint8)))
        with self.assertRaises(TimeoutError):
            camera.read(timeout=.01)
        camera.frames.put((time.monotonic(), np.zeros((10, 10, 3), np.uint8)))
        self.assertEqual(camera.read().shape, (10, 10, 3))


if __name__ == '__main__':
    unittest.main()

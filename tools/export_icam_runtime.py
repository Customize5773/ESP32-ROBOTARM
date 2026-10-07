"""Export current notebook runtime without Colab, training, or loading model weights.

python tools/export_icam_runtime.py --output artifacts/icam_runtime_far_patch
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    'runtime-module': 'icam_color_runtime.py',
    'mapping-module': 'robot_mapping.py',
    'stage-module': 'stage_coordinator.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts'/'icam_runtime_far_patch')
    args = parser.parse_args()
    destination = args.output.resolve()
    archive = destination.with_name(destination.name+'.zip')
    if destination.exists() or archive.exists():
        parser.error('Pilih folder output baru; file yang sudah ada tidak ditimpa.')
    notebook = ROOT/'Train_arm_yolo1.ipynb'
    cells = {c['id']: ''.join(c['source']) for c in json.loads(notebook.read_text(encoding='utf-8'))['cells']}
    files = {}
    for cell_id, filename in MODULES.items():
        first, source = cells[cell_id].split('\n', 1)
        if first.strip() != '%%writefile '+filename:
            raise ValueError('Header cell tidak cocok: '+cell_id)
        ast.parse(source, filename=filename)
        files[filename] = source.encode('utf-8')
    files['mat_roi.example.json'] = (json.dumps(dict(frame_size_wh=None, points_px=None), indent=2)+'\n').encode()
    files['README.md'] = (ROOT/'ICAM-FAR-DETECTION.md').read_bytes()
    files['runtime_patch.json'] = (json.dumps(dict(
        source_notebook='Train_arm_yolo1.ipynb',
        notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
        contains_model_weights=False,
        default_with_roi='runtime_config.inference, fallback roi-crop', optional_modes=['tiled','small'],
        coordinates='original_camera_pixels; XY mm only with measured calibration',
        motor_transport_enabled=False), indent=2)+'\n').encode()
    files['SHA256SUMS.json'] = (json.dumps({name: hashlib.sha256(data).hexdigest()
        for name, data in files.items()}, indent=2)+'\n').encode()
    destination.mkdir(parents=True)
    for name, data in files.items():
        (destination/name).write_bytes(data)
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    print(archive)


if __name__ == '__main__':
    main()

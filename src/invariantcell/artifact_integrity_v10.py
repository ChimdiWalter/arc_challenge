"""Verify persisted scientific outputs against their saved execution receipt."""
from pathlib import Path
import json
from .io import file_hash

def verify_run(directory):
    directory=Path(directory).resolve();checked={}
    def bound(name,expected):
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path)!=expected:
            raise ValueError('Run artifact missing, escaping, or modified: '+name)
        checked[name]=expected
    receipt=directory/'receipt.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        for name,sha in value.get('artifact_sha256',{}).items():bound(name,sha)
        if 'result_sha256' in value:bound('result.json',value['result_sha256'])
        if 'demo_sha256' in value:bound('demo.json',value['demo_sha256'])
    prediction=directory/'prediction_receipt.json'
    if prediction.exists():
        value=json.loads(prediction.read_text());bound('predicted_instances.npy',value['prediction_sha256'])
        result=json.loads((directory/'results.json').read_text())
        if result['prediction_sha256']!=value['prediction_sha256'] or result['package_sha256']!=value['package_sha256']:
            raise ValueError('Measurement result bound to different inputs/predictions')
    return checked

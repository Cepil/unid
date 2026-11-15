"""Reuse T37 R artifacts without retraining; verify against saved Validation probabilities.

Train/Validation metadata and Validation probabilities remain on the server.
Does not access Test. The separate test.py performs authorised inference.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import os
import shutil

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import base_model as B
import base_epair as E
import model as M
import embedding_cache as CACHE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source_dir',type=Path,required=True)
    p.add_argument('--data_dir',type=Path,required=True)
    p.add_argument('--cache_dir',type=Path,default=Path.home()/'unid_work/hv2_cache')
    p.add_argument('--output_dir',type=Path,default=Path(__file__).parent/'weights')
    args=p.parse_args()
    if socket.gethostname().split('.')[0]!=os.environ.get('UNID_EXPECTED_HOST','da-perfect28'):
        raise RuntimeError('Use the competition server only')
    if args.output_dir.exists():
        raise FileExistsError('Fresh output directory required')
    metrics=json.loads((args.source_dir/'metrics.json').read_text())
    if metrics['config']['experiment']!='T37':
        raise ValueError('Expected T37 source')
    for name,old in [('base_epair.py','epa_model.py'),('base_model.py','base_model.py'),
                     ('baseline_v7.py','baseline_v7.py'),('robust_sensor_scaling.py','robust_sensor_scaling.py')]:
        if sha(Path(__file__).parent/name)!=metrics['config']['sources'][old]:
            raise ValueError('Inference source differs from T37')
    csv=args.data_dir/'validation/validation.csv'
    if sha(csv)!=metrics['config']['data_csv_sha256']['validation']:
        raise ValueError('Validation CSV differs from T37')
    B.seed_all(42)
    reference=torch.load(args.source_dir/'reference_model.pt',map_location='cpu',weights_only=True)
    heads=torch.load(args.source_dir/'robust_heads.pt',map_location='cpu',weights_only=True)
    if importlib.metadata.version('scikit-learn')!=reference['environment_versions']['scikit-learn']:
        raise RuntimeError('Use trained scikit-learn version')
    selected=M.select_r(reference,heads)
    trees=joblib.load(args.source_dir/'reference_tree.joblib')
    frame=M.read_csv(csv)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=B.encoder(pretrained=False)
    backbone.load_state_dict(reference['encoder'])
    raw=CACHE.get_embeddings(frame,csv,[args.data_dir/'validation',args.data_dir],
                             args.cache_dir/'validation_resnet18.pt',backbone.to(device),device)
    result=M.predict(frame,raw,selected,trees,device)
    with np.load(args.source_dir/'validation_private.npz') as stored:
        expected=stored['R']
    if result.shape!=expected.shape:
        raise ValueError('Validation probability shape mismatch')
    maximum=float(np.max(np.abs(result-expected)))
    changed=int((result.argmax(1)!=expected.argmax(1)).sum())
    np.testing.assert_allclose(result,expected,rtol=0,atol=1e-12)
    if changed:
        raise ValueError('Converted artifact changes predictions')
    score=M.metrics(B.labels(frame),result)
    if not np.isclose(score['macro_f1'],metrics['arms']['R']['validation']['macro_f1'],rtol=0,atol=1e-12):
        raise ValueError('Validation metric mismatch')
    args.output_dir.mkdir(parents=True)
    torch.save(selected,args.output_dir/'model.pt')
    shutil.copyfile(args.source_dir/'reference_tree.joblib',args.output_dir/'tree.joblib')
    manifest={'model_family':M.MODEL_FAMILY,'format_version':M.FORMAT_VERSION,'trained_on':'provided Train only',
              'origin':'T37 R; no retraining during conversion','sensor_transform':M.TRANSFORM,
              'environment_versions':selected['environment_versions'],
              'files':{name:sha(args.output_dir/name) for name in ('model.pt','tree.joblib')},
              'sources':{path.name:sha(path) for path in Path(__file__).parent.glob('*.py')}}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    reloaded=torch.load(args.output_dir/'model.pt',map_location='cpu',weights_only=True)
    replay=M.predict(frame,raw,reloaded,joblib.load(args.output_dir/'tree.joblib'),device)
    np.testing.assert_allclose(replay,expected,rtol=0,atol=1e-12)
    (args.output_dir/'validation_metrics.json').write_text(json.dumps(
        {'metrics':score,'max_abs_prob_diff_vs_T37_R':maximum,'changed_labels':changed,
         'validation_note':'Repeatedly consulted during research; not independent holdout'},indent=2))
    print(f'PASS: T37 R conversion and reload; max_probability_diff={maximum:.3g}, changed_labels={changed}',flush=True)
    print(f'Validation Macro F1={score["macro_f1"]:.6f}',flush=True)
    print('Weights saved. No retraining or Test access during conversion.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()

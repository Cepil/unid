"""Local checks only: CSV schema/IDs/order and repeat/subset/order invariance.

Never prints sample IDs, sensor values, labels or images. Does not fit a model.
On Seraph, run only within an allocated compute node; competition servers need no Slurm.
"""
import argparse
import os
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import pandas as pd

def read_submission(path):
    data=pd.read_csv(path,dtype={'sample_id':'string'})
    if list(data.columns)!=['sample_id','label']:
        raise ValueError('Submission columns must be sample_id,label in that order')
    if data.empty or data.isna().any().any() or not data.sample_id.is_unique:
        raise ValueError('Submission has missing/duplicate IDs or missing values')
    if not data.label.isin(range(1,7)).all():
        raise ValueError('Submission labels must be integers 1..6')
    if not pd.api.types.is_integer_dtype(data.label):
        raise ValueError('Submission label column must have integer dtype')
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--image_root',type=Path,required=True)
    p.add_argument('--submission',type=Path,required=True)
    p.add_argument('--sample_submission',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'weights/hybrid.pt')
    p.add_argument('--temp_dir',type=Path,required=True,help='Private compute-node local directory')
    args=p.parse_args()
    host=socket.gethostname().split('.')[0]
    if host.startswith(('moana-','ariel-','aurora-')):
        if host.endswith(('-master','-login')) or not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('On Seraph, use an allocated compute node')
    frame=pd.read_csv(args.input,dtype={'sample_id':'string'})
    template=pd.read_csv(args.sample_submission,dtype={'sample_id':'string'})
    full=read_submission(args.submission)
    for name,ids in [('Test',frame.sample_id),('template',template.sample_id)]:
        if ids.isna().any() or not ids.is_unique or set(ids)!=set(full.sample_id):
            raise ValueError(f'{name}/submission ID mismatch')
    if full.sample_id.tolist()!=template.sample_id.tolist():
        raise ValueError('Submission row order differs from sample_submission.csv; reorder by ID before submission')
    print('PASS: submission columns, labels, IDs and official template order',flush=True)
    # Spread checks across the full input; no label-dependent sample selection.
    count=min(64,len(frame))
    positions=sorted({i*(len(frame)-1)//max(1,count-1) for i in range(count)})
    subset=frame.iloc[positions].copy()
    reference=full.set_index('sample_id')['label']
    args.temp_dir.mkdir(parents=True,exist_ok=True)
    runner=Path(__file__).resolve().parent/'test.py'
    with tempfile.TemporaryDirectory(prefix='verify-',dir=args.temp_dir) as directory:
        directory=Path(directory)
        for name,part in [('subset',subset),('repeat',subset),('reversed',subset.iloc[::-1])]:
            csv=directory/f'{name}.csv'
            output=directory/f'{name}_pred.csv'
            part.to_csv(csv,index=False)
            subprocess.run([sys.executable,str(runner),'--input',str(csv),
                            '--image_root',str(args.image_root),'--model',str(args.model),
                            '--output',str(output),'--batch_size','17','--workers','0'],check=True)
            pred=read_submission(output)
            if pred.sample_id.tolist()!=part.sample_id.tolist():
                raise ValueError(f'{name}: inference changed input row order')
            expected=reference.loc[pred.sample_id].to_numpy()
            if not (pred.label.to_numpy()==expected).all():
                raise ValueError(f'{name}: predictions differ from full input; do not submit yet')
            print(f'PASS: {name} input prediction consistency',flush=True)
    print('All checks passed. Partial-input checks cover a deterministic sample, not every possible subset.',flush=True)

if __name__=='__main__':
    main()

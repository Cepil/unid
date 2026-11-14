"""Pure inference with saved hybrid artifacts; no pretrained downloads or fitting."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import base_model as B
import model as H

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--image_root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'weights/hybrid.pt')
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    B.seed_all(42)
    bundle=torch.load(args.model,map_location='cpu',weights_only=True)
    if bundle['format_version']!=2:
        raise ValueError('Unsupported checkpoint format')
    # Load only the artifacts produced by our own training, never an untrusted joblib file.
    tree_file=Path(bundle['tree_file'])
    if tree_file.name!=str(tree_file):
        raise ValueError('Tree artifact must be a sibling filename')
    trees=joblib.load(args.model.parent/tree_file)
    frame=B.read_csv(args.input)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=B.encoder(pretrained=False)
    backbone.load_state_dict(bundle['encoder'])
    embeddings=B.extract(frame,[args.image_root,args.image_root.parent],backbone.to(device),
                          device,args.batch_size,args.workers)
    probs=H.predict_from_embeddings(frame,embeddings,bundle,trees,device)['fusion_hybrid']
    labels=probs.argmax(1)+1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'sample_id':frame.sample_id,'label':labels}).to_csv(args.output,index=False)
    print('Hybrid submission saved in input CSV row order.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()

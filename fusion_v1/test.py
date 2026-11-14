"""Inference only. No pretrained download, labels or temporal context required."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import model as M

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--image_root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'weights/fusion.pt')
    p.add_argument('--batch_size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    M.seed_all()
    bundle=torch.load(args.model,map_location='cpu',weights_only=True)
    if bundle['format_version']!=1:
        raise ValueError('Unknown checkpoint format')
    frame=M.read_csv(args.input)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=M.encoder()
    backbone.load_state_dict(bundle['encoder'])
    embeddings=M.extract(frame,[args.image_root,args.image_root.parent],backbone.to(device),
                         device,args.batch_size,args.workers)
    embeddings=(embeddings-bundle['image_mean'])/bundle['image_std']
    prep={k:np.asarray(v) for k,v in bundle['preprocessing'].items()}
    sensors=torch.from_numpy(M.apply_preprocessing(frame,prep))
    net=M.Fusion().to(device)
    net.load_state_dict(bundle['fusion'])
    net.eval()
    pred=[]
    with torch.inference_mode():
        for start in range(0,len(frame),256):
            pred.append(net(embeddings[start:start+256].to(device),
                            sensors[start:start+256].to(device)).argmax(1).cpu())
    labels=torch.cat(pred).numpy()+1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'sample_id':frame.sample_id,'label':labels}).to_csv(args.output,index=False)
    print('Submission saved in input CSV row order.',flush=True)

if __name__=='__main__':
    main()

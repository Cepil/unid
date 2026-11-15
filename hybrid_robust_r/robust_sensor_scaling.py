"""Candidate numerical transform for MLP sensor inputs, not fitted to real data here.

Use fold Train only to compute parameters. Keep image/tree preprocessing intact.
No labels, I/O, experiment normalisation or whole-sequence statistics are used.
"""
import numpy as np


def fit_robust_scaling(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)==0:
        raise ValueError('Expected nonempty 2D feature matrix')
    median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
    iqr=np.array([np.subtract(*np.quantile(c[np.isfinite(c)],[.75,.25])) if np.isfinite(c).any() else 1. for c in x.T])
    iqr[iqr<1e-8]=1.
    return {'median':median.tolist(),'iqr':iqr.tolist()}


def apply_robust_scaling(x,params,mode):
    x=np.asarray(x,dtype=float)
    median,iqr=np.asarray(params['median']),np.asarray(params['iqr'])
    if x.ndim!=2 or x.shape[1]!=len(median) or len(iqr)!=len(median) or not (iqr>0).all():
        raise ValueError('Invalid feature/parameter schema')
    z=(np.where(np.isfinite(x),x,median)-median)/iqr
    if mode=='robust_clip':
        out=z.clip(-20,20)
    elif mode=='robust_asinh':
        out=np.arcsinh(z)
    else:
        raise ValueError('Unknown predeclared transform')
    return out.astype(np.float32)


def synthetic_check():
    train=np.array([[0.,2.,np.nan],[1.,2.,np.nan],[2.,2.,np.nan],[3.,2.,np.nan],[4.,2.,np.nan]])
    before=train.copy()
    params=fit_robust_scaling(train)
    assert params=={'median':[2.,2.,0.],'iqr':[2.,1.,1.]}
    heldout=np.array([[-100.,2.,np.inf],[2.,2.,np.nan],[100.,2.,0.]])
    for mode in ('robust_clip','robust_asinh'):
        got=apply_robust_scaling(heldout,params,mode)
        assert np.isfinite(got).all() and np.array_equal(got[:,1:],np.zeros((3,2)))
        assert got[0,0]<got[1,0]<got[2,0]
        np.testing.assert_array_equal(got[::-1],apply_robust_scaling(heldout[::-1],params,mode))
    assert apply_robust_scaling(heldout,params,'robust_clip')[0,0]==-20
    np.testing.assert_allclose(apply_robust_scaling(heldout,params,'robust_asinh')[:,0],np.arcsinh([-51.,0.,49.]),rtol=1e-6)
    np.testing.assert_array_equal(train,before)
    assert params=={'median':[2.,2.,0.],'iqr':[2.,1.,1.]}
    print('PASS: robust transform formula, constant/all-missing columns, heldout application, order invariance, no mutation')


if __name__=='__main__':
    synthetic_check()

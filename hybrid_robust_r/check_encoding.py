"""Synthetic numerical encoding check only; no ML imports or inference."""
import ast
from pathlib import Path
import numpy as np
import robust_sensor_scaling as RS


def main():
    root=Path(__file__).parent
    def load(path,name,scope):
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),name,'exec'),scope)
        return scope[name]
    encoded=load(root/'model.py','encoded_preprocessing',{'np':np})
    apply=load(root/'base_epair.py','apply_prep',{'np':np})
    rng=np.random.default_rng(17)
    train=rng.normal(size=(100,9))
    train[:,1]=0.
    train[:,8]=np.nan
    params=RS.fit_robust_scaling(train)
    eval=rng.normal(size=(80,9))*100
    eval[0,0]=np.inf;eval[1,0]=-np.inf;eval[2,0]=np.nan
    np.testing.assert_array_equal(RS.apply_robust_scaling(eval,params,'robust_clip'),apply(eval,encoded(params)))
    RS.synthetic_check()
    print('PASS: existing inference preprocessing represents R exactly (including missing/nonfinite values)')


if __name__=='__main__':
    main()

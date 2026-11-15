"""EPAIR with T37 R heads and Train median/IQR sensor scaling, clip +/-20.

The existing affine preprocessing format can represent RobustScaler exactly:
imputation median = Train median, centering mean field = Train median,
scale std field = Train IQR (constant columns use 1). No inference changes.
"""
import numpy as np
import base_epair as E
import base_model as B
import robust_sensor_scaling as RS

FORMAT_VERSION=17
MODEL_FAMILY='hybrid_robust_r'
BASELINE_SHA256=E.BASELINE_SHA256
TRANSFORM='train_median_iqr_clip20'
read_csv=E.read_csv
metrics=E.metrics


def encoded_preprocessing(params):
    median,iqr=np.asarray(params['median'],dtype=float),np.asarray(params['iqr'],dtype=float)
    if median.shape!=(9,) or iqr.shape!=(9,) or not np.isfinite(median).all() or not np.isfinite(iqr).all() or not (iqr>0).all():
        raise ValueError('Invalid R preprocessing')
    return {'median':median.tolist(),'mean':median.tolist(),'std':iqr.tolist()}


def select_r(reference,heads):
    if reference.get('model_family')!=E.MODEL_FAMILY or reference.get('format_version')!=E.FORMAT_VERSION:
        raise ValueError('Expected canonical EPAIR reference')
    if heads['modes'].get('R')!='robust_clip' or heads['seeds']!=list(E.SEEDS) or heads['epochs']!=reference['epochs']:
        raise ValueError('R head recipe mismatch')
    if len(heads['states']['R'])!=len(E.SEEDS):
        raise ValueError('Incomplete R seeds')
    return {**reference,'format_version':FORMAT_VERSION,'model_family':MODEL_FAMILY,
            'sensor_transform':TRANSFORM,'prep_current':encoded_preprocessing(heads['prep_sensor']),
            'states':heads['states']['R'],'selection_note':'T37 R; exploratory Public check authorised after CV rule failure'}


def predict(frame,raw_images,bundle,trees,device,return_components=False):
    if bundle.get('model_family')!=MODEL_FAMILY or bundle.get('format_version')!=FORMAT_VERSION or bundle.get('sensor_transform')!=TRANSFORM:
        raise ValueError('Expected hybrid_robust_r artifact')
    p=bundle['prep_current']
    if p['mean']!=p['median']:
        raise ValueError('R centering must use Train median')
    encoded_preprocessing({'median':p['median'],'iqr':p['std']})
    internal={**bundle,'format_version':E.FORMAT_VERSION,'model_family':E.MODEL_FAMILY}
    return E.predict(frame,raw_images,internal,trees,device,return_components=return_components)


def fit(frame,raw_images,device,epochs=E.EPOCHS,tree_iterations=None,pca_components=32):
    """Reproduce R on Train only; shared canonical trees, then three R heads."""
    from mlp_training import train_mlp
    reference,trees=E.fit(frame,raw_images,device,epochs=epochs,tree_iterations=tree_iterations,pca_components=pca_components)
    raw=B.numeric_features(frame)
    prep=RS.fit_robust_scaling(raw)
    image=(raw_images-reference['image_mean'])/reference['image_std']
    heads={'prep_sensor':prep,'modes':{'R':'robust_clip'},'seeds':list(E.SEEDS),'epochs':epochs,
           'states':{'R':train_mlp(image,RS.apply_robust_scaling(raw,prep,'robust_clip'),B.labels(frame),device,epochs)}}
    return select_r(reference,heads),trees

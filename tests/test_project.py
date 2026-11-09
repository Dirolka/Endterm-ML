
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from src.project import *

@pytest.fixture(scope='module')
def data():
    df=load_data(); return (df,*load_split(df))

def test_dataset_loading_and_database(data):
    df,a,b=data
    assert len(df)==1250 and df.id.is_unique
    assert set(df.grade)==set(LABELS)
    assert validate_database(df)['csv_db_all_raw_fields_match']

def test_saved_split_exact_order_and_coverage(data):
    df,a,b=data
    assert len(a)==1000 and len(b)==250 and not set(a.id)&set(b.id)
    assert set(a.id)|set(b.id)==set(df.id)
    assert a.id.tolist()==pd.read_csv(ORIGINAL/'train_ids.csv',dtype=str).id.tolist()

@pytest.mark.parametrize('defect',['duplicate','overlap','missing','unknown'])
def test_bad_splits_rejected(data,defect):
    df,a,b=data; aa=a.id.tolist(); bb=b.id.tolist()
    if defect=='duplicate': aa[1]=aa[0]
    if defect=='overlap': bb[0]=aa[0]
    if defect=='missing': aa.pop()
    if defect=='unknown': aa[0]='nonexistent'
    with pytest.raises(ValueError): validate_split(df,aa,bb)

def test_scenario_b_excludes_proxies_and_unknown_schedule(data):
    df,a,b=data; p=preprocess().fit(a)
    names=p.get_feature_names_out()
    assert not any(any(s in n for s in ['title','experience','name__','grade','id__']) for n in names)
    changed=b.copy(); changed['name']='Junior Senior'; changed['experience_id']='moreThan6'; changed['experience']='changed'
    np.testing.assert_array_equal(p.transform(b),p.transform(changed))
    changed['schedule']='unseen'; assert np.isfinite(p.transform(changed)).all()

def test_fold_imputer_learns_training_only(data):
    df,a,b=data
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=SEED).split(a,a.grade):
        p=preprocess().fit(a.iloc[tr]); im=p.named_transformers_['numeric'].named_steps['impute']
        np.testing.assert_allclose(im.statistics_,a.iloc[tr][NUM].median().to_numpy())
        old=im.statistics_.copy(); shifted=a.iloc[va].copy(); shifted['salary_mean']=1e15
        p.transform(shifted); np.testing.assert_array_equal(old,im.statistics_)

def test_metric_calculation():
    m=metrics(['Junior','Middle','Senior'],['Junior','Junior','Senior'])
    assert m['test_accuracy']==pytest.approx(2/3)
    assert m['test_macro_f1']==pytest.approx((2/3+0+1)/3)

def test_saved_models_predict_compatibly(data):
    df,a,b=data
    files=list((ROOT/'results/models').glob('*.joblib')); assert len(files)==6
    for f in files:
        m=joblib.load(f); p=m.predict(b)
        assert len(p)==250 and set(p)<=set(LABELS)
        assert np.isfinite(m.predict_proba(b)).all()

def test_original_checksums():
    assert original_hashes()==json.loads((ROOT/'results/original_manifest.json').read_text())

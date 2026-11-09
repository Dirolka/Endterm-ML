
from pathlib import Path
import hashlib, json, re, sqlite3, importlib.util
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'data/raw'
SEED = 42
LABELS = ['Junior', 'Middle', 'Senior']
SKILLS = ['python','go','fastapi','postgresql','docker','git','linux','kubernetes','redis','kafka','ci/cd','microservices']
NUM = ['salary_mean','is_salary_missing','desc_length','desc_words_count','num_skills']
BINARY = ['skill_'+s.replace('/','_') for s in SKILLS]

def historical_module():
    spec = importlib.util.spec_from_file_location('midterm_readonly', ROOT/'src/_midterm_reference.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_data():
    df = pd.read_csv(ORIGINAL/'hh_vacancies_raw_snapshot.csv', dtype={'id':str})
    required = {'id','name','experience','experience_id','salary_from','salary_to','schedule','key_skills','description'}
    if not required.issubset(df):
        raise ValueError('Missing required raw fields')
    if df.id.isna().any() or not df.id.is_unique:
        raise ValueError('Missing or duplicate raw IDs')
    
    old = historical_module()
    df['grade'] = df.apply(old.assign_grade, axis=1)
    clean = df.description.map(old.clean_html)
    df['desc_length'] = clean.str.len()
    df['desc_words_count'] = clean.map(lambda x:len(x.split()))
    df['salary_mean'] = df[['salary_from','salary_to']].mean(axis=1)
    df['is_salary_missing'] = df[['salary_from','salary_to']].isna().all(axis=1).astype(int)
    skills = df.key_skills.map(json.loads)
    df['num_skills'] = skills.map(len)
    for skill,col in zip(SKILLS,BINARY):
        
        df[col] = skills.map(lambda ss:int(any(skill in str(s).lower() for s in ss)))
    return df.set_index('id', drop=False)

def validate_split(df, train_ids, test_ids, expected=(1000,250)):
    a,b = list(train_ids),list(test_ids)
    if (len(a),len(b)) != expected:
        raise ValueError('Unexpected saved split counts')
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise ValueError('Duplicate split IDs')
    if set(a)&set(b):
        raise ValueError('Overlapping split')
    if set(a)|set(b) != set(df.index):
        raise ValueError('Split does not cover exactly the dataset')
    return df.loc[a].copy(),df.loc[b].copy()

def load_split(df):
    return validate_split(df,*[pd.read_csv(ORIGINAL/f'{s}_ids.csv',dtype=str).id for s in ['train','test']])

def preprocess(scenario='B', scale_all=False):
    num = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    parts = [('numeric',num,NUM),('skills','passthrough',BINARY),('schedule',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['schedule'])]
    if scenario=='A':
        parts += [('title',TfidfVectorizer(max_features=40),'name'),('experience',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['experience_id'])]
    ct = ColumnTransformer(parts,remainder='drop',sparse_threshold=0)
    return Pipeline([('columns',ct),('scale_all',StandardScaler())]) if scale_all else ct

def pipeline(model,scenario='B',scale_all=False):
    return Pipeline([('prep',preprocess(scenario,scale_all)),('model',model)])

def metrics(y,p):
    return {'test_accuracy':accuracy_score(y,p),'test_macro_f1':f1_score(y,p,average='macro',zero_division=0),
            'test_precision':precision_score(y,p,average='macro',zero_division=0),'test_recall':recall_score(y,p,average='macro',zero_division=0)}

def original_hashes():
    paths = sorted(p for p in ORIGINAL.iterdir() if p.is_file()) + [ROOT/'src/_midterm_reference.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def validate_database(df):
    with sqlite3.connect(f'file:{ORIGINAL / "hh_vacancies_raw.db"}?mode=ro',uri=True) as con:
        db = pd.read_sql_query('SELECT * FROM raw_vacancies',con).astype({'id':str}).set_index('id',drop=False)
    columns = list(pd.read_csv(ORIGINAL/'hh_vacancies_raw_snapshot.csv',nrows=0).columns)
    pd.testing.assert_frame_equal(df.loc[sorted(df.index),columns].reset_index(drop=True), db.loc[sorted(db.index),columns].reset_index(drop=True),check_dtype=False)
    return {'rows':len(db),'csv_db_all_raw_fields_match':True}

"""Real Djinni benchmark. No generated vacancies and no title-derived labels."""
from pathlib import Path
import hashlib, json, re, warnings
import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/real'
LABELS=['0–1 years','2–3 years','5+ years']
MAPPING={'no_exp':LABELS[0],'1y':LABELS[0],'2y':LABELS[1],'3y':LABELS[1],'5y':LABELS[2]}
TECH=['python','java','javascript','typescript','sql','postgresql','mysql','docker','kubernetes','aws','azure','git','linux','react','angular','node','django','fastapi','redis','kafka','spark','scala','golang','php','c++','c#']
NUM=['desc_words','desc_chars','skill_count']+['tech_'+str(i) for i in range(len(TECH))]
CAT=['Primary Keyword','English Level']
SOURCE='https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english'
REV='b56a6054c10f1266a141e37c8df66c79ff2863af'

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))

def prepare():
    raw=ROOT/'data/real/djinni_jobs.parquet'
    df=pd.DataFrame(pl.read_parquet(raw).to_dicts())
    audit={'source':SOURCE,'revision':REV,'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'raw_rows':len(df),'raw_experience_counts':df['Exp Years'].value_counts(dropna=False).to_dict(),'mapping':MAPPING}
    df=df.dropna(subset=['id','Long Description','Company Name','Exp Years']).copy()
    df=df[df['Exp Years'].isin(MAPPING)].copy()
    df['company_group']=df['Company Name'].str.lower().str.replace(r'[^\w]+','',regex=True)
    df=df[df.company_group.ne('')].copy()
    df['description_key']=df['Long Description'].str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
    df=df.sort_values('id').drop_duplicates('id').drop_duplicates('description_key').copy()
    # Identical normalized descriptions are removed globally before any sampling/split.
    audit['after_exact_dedup']=len(df)
    df=df.sample(n=min(12000,len(df)),random_state=42).sort_values('id').reset_index(drop=True)
    df['target']=df['Exp Years'].map(MAPPING)
    text=df['Long Description'].str.lower()
    # Title and raw description are excluded. Only fixed tech flags and aggregate sizes enter.
    df['desc_words']=text.str.split().str.len()
    df['desc_chars']=text.str.len()
    for i,tech in enumerate(TECH):
        df['tech_'+str(i)]=text.str.contains(r'(?<!\w)'+re.escape(tech)+r'(?!\w)',regex=True).astype(int)
    df['skill_count']=df[['tech_'+str(i) for i in range(len(TECH))]].sum(axis=1)
    df[CAT]=df[CAT].fillna('Unknown')
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,df.target,df.company_group))
    train,test=df.iloc[a].copy(),df.iloc[b].copy()
    assert not set(train.company_group)&set(test.company_group)
    assert not set(train.description_key)&set(test.description_key)
    assert set(train.target)==set(test.target)==set(LABELS)
    audit.update(sample_rows=len(df),train_rows=len(train),test_rows=len(test),train_companies=train.company_group.nunique(),test_companies=test.company_group.nunique(),company_overlap=0,exact_description_overlap=0,class_counts=df.target.value_counts().to_dict(),sample_seed=42,sample_reason='Bounded compute; uniform sample of deduplicated real records',features=NUM+CAT)
    for name,sub in [('train',train),('test',test)]:
        sub[['id','Company Name','company_group','target']].to_csv(ROOT/f'data/real/{name}_ids.csv',index=False)
    df.drop(columns=['description_key']).to_csv(ROOT/'data/real/benchmark.csv',index=False)
    dump(OUT/'data_audit.json',audit)
    return df,train,test,audit

def preprocessing():
    return ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)],remainder='drop')

def run():
    for p in [OUT,OUT/'tables',OUT/'figures',OUT/'models']:p.mkdir(parents=True,exist_ok=True)
    df,train,test,audit=prepare()
    print('DATA',audit['raw_rows'],len(train),len(test),flush=True)
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    folds=list(splitter.split(train,train.target,train.company_group))
    for a,b in folds:
        assert not set(train.iloc[a].company_group)&set(train.iloc[b].company_group)
    specs={
      'Dummy':(DummyClassifier(strategy='most_frequent'),{}),
      'Logistic Regression':(LogisticRegression(max_iter=1500,C=1),{}),
      'KNN':(KNeighborsClassifier(n_neighbors=15,weights='distance'),{}),
      'Random Forest':(RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=1),{'model__max_depth':[12,None],'model__min_samples_leaf':[2,8]}),
      'Boosting':(HistGradientBoostingClassifier(max_iter=150,early_stopping=False,random_state=42),{'model__max_leaf_nodes':[7,15],'model__l2_regularization':[1,5]}),
      'MLP':(MLPClassifier(hidden_layer_sizes=(32,),max_iter=180,early_stopping=False,random_state=42),{'model__alpha':[.1,1.]})}
    models={};selection={}; searches={}
    for name,(model,grid) in specs.items():
        print('CV',name,flush=True)
        p=Pipeline([('prep',preprocessing()),('model',model)])
        search=GridSearchCV(p,grid,cv=folds,scoring='f1_macro',n_jobs=1,return_train_score=True,error_score='raise')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); search.fit(train,train.target)
        models[name]=search.best_estimator_
        selection[name]={'cv_mean':search.best_score_,'cv_std':search.cv_results_['std_test_score'][search.best_index_],'params':search.best_params_,'warnings':sorted(set(str(w.message) for w in caught))}
        searches[name]=grid
        pd.DataFrame(search.cv_results_).to_csv(OUT/'tables'/f'{name.replace(" ","_")}_cv.csv',index=False)
        print(name,selection[name]['cv_mean'],flush=True)
    preferred=max(selection,key=lambda k:selection[k]['cv_mean'])
    dump(OUT/'selection.json',{'preferred_model':preferred,'criterion':'Highest mean 5-fold employer-grouped training CV macro F1','models':selection,'search_spaces':searches})
    # Freeze selection before any held-out predictions.
    rows=[];predictions=test[['id','Position','Company Name','Exp Years','target','Primary Keyword','English Level','desc_words']].copy()
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    reports={}
    for (name,m),ax in zip(models.items(),axs.flat):
        pred=m.predict(test); predictions[name]=pred
        r=classification_report(test.target,pred,labels=LABELS,output_dict=True,zero_division=0);reports[name]=r
        rows.append({'model':name,**{k:v for k,v in selection[name].items() if k in ['cv_mean','cv_std']},'test_macro_f1':f1_score(test.target,pred,average='macro'),'test_accuracy':accuracy_score(test.target,pred),'errors':int((pred!=test.target).sum())})
        dump(OUT/'tables'/f'{name.replace(" ","_")}_report.json',r)
        cm=confusion_matrix(test.target,pred,labels=LABELS)
        pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(OUT/'tables'/f'{name.replace(" ","_")}_confusion.csv')
        ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,colorbar=False);ax.set_title(name)
        joblib.dump(m,OUT/'models'/f'{name.replace(" ","_")}.joblib')
    fig.tight_layout();fig.savefig(OUT/'figures/confusion_matrices.png',dpi=150);plt.close(fig)
    comp=pd.DataFrame(rows).sort_values('cv_mean',ascending=False);comp.to_csv(OUT/'tables/model_comparison.csv',index=False)
    predictions.to_csv(OUT/'tables/test_predictions.csv',index=False)
    errors=predictions[predictions[preferred]!=predictions.target].copy();errors['predicted']=errors[preferred];errors.to_csv(OUT/'tables/preferred_errors.csv',index=False)
    # Independent training-only grouped diagnostic for MLP.
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=43).split(train,groups=train.company_group))
    prep=preprocessing();xa=prep.fit_transform(train.iloc[a]);xb=prep.transform(train.iloc[b])
    net=clone(models['MLP'].named_steps['model']);curve=[]
    for epoch in range(1,101):
        net.partial_fit(xa,train.iloc[a].target,classes=np.array(LABELS))
        if epoch==1 or epoch%5==0:
            curve.append({'epoch':epoch,'loss':net.loss_,'train_f1':f1_score(train.iloc[a].target,net.predict(xa),average='macro'),'validation_f1':f1_score(train.iloc[b].target,net.predict(xb),average='macro')})
    pd.DataFrame(curve).to_csv(OUT/'tables/mlp_learning.csv',index=False)
    final=models['MLP'].named_steps['model']
    dump(OUT/'mlp.json',{'input_features':xa.shape[1],'hidden':[32],'output':3,'iterations':final.n_iter_,'max_iter':final.max_iter,'hit_cap':final.n_iter_>=final.max_iter,'alpha':final.alpha,'last_diagnostic':curve[-1],'diagnostic_fit_rows':len(a),'diagnostic_validation_rows':len(b)})
    # PCA only on train, labels never supplied to fit.
    prep=preprocessing();x=prep.fit_transform(train);pc=PCA().fit(x);z=pc.transform(x)
    pd.DataFrame({'component':np.arange(1,len(pc.explained_variance_ratio_)+1),'variance':pc.explained_variance_ratio_,'cumulative':pc.explained_variance_ratio_.cumsum()}).to_csv(OUT/'tables/pca_variance.csv',index=False)
    pd.DataFrame({'pc1':z[:,0],'pc2':z[:,1],'target':train.target.values}).to_csv(OUT/'tables/pca_projection.csv',index=False)
    pd.DataFrame(pc.components_[:2].T,index=prep.get_feature_names_out(),columns=['PC1','PC2']).to_csv(OUT/'tables/pca_loadings.csv')
    dump(OUT/'pca.json',{'features':x.shape[1],'first_two':pc.explained_variance_ratio_[:2].sum(),'components95':int(np.searchsorted(pc.explained_variance_ratio_.cumsum(),.95)+1)})
    # Predictions must ignore labels/title/company; verify by perturbation.
    changed=test.copy();changed['Exp Years']='made_up';changed['target']='made_up';changed['Position']='Senior';changed['Company Name']='Other'
    assert np.array_equal(models[preferred].predict(test),models[preferred].predict(changed))
    dump(OUT/'verification.json',{'company_overlap':0,'exact_description_overlap':0,'cv_group_overlap':0,'proxy_perturbation_invariant':True,'prediction_rows':len(predictions),'preferred_errors':len(errors),'source_sha256':audit['sha256']})
    print(comp.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()


import json, warnings, sys, platform
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from src.project import *

OUT = ROOT/'results'
T = OUT/'tables'
F = OUT/'figures'
def savefig(name):
    plt.tight_layout(); plt.savefig(F/f'{name}.png',dpi=170); plt.close()
def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=str))

def reproduce(train,test):
    
    df = pd.concat([train,test]); old = historical_module()
    all_df = old.prepare_dataset(str(ORIGINAL/'hh_vacancies_raw_snapshot.csv')).astype({'id':str}).set_index('id')
    a,b = all_df.loc[train.index],all_df.loc[test.index]
    num = ['salary_mean_imputed']+NUM[1:]
    flags = [c for c in all_df if c.startswith('skill_') or c.startswith('schedule_')]
    scaler=StandardScaler(); xa=np.hstack([scaler.fit_transform(a[num]),a[flags].values]); xb=np.hstack([scaler.transform(b[num]),b[flags].values])
    rows=[]
    for scenario in ['A','B']:
        xtr,xte=xa,xb
        if scenario=='A':
            tf=TfidfVectorizer(max_features=40); enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
            xtr=np.hstack([xa,tf.fit_transform(a.name).toarray(),enc.fit_transform(a[['experience_id']])])
            xte=np.hstack([xb,tf.transform(b.name).toarray(),enc.transform(b[['experience_id']])])
        for name,m in [('Logistic Regression',LogisticRegression(max_iter=1000,random_state=SEED)),('KNN',KNeighborsClassifier(n_neighbors=5 if scenario=='A' else 7,weights='distance'))]:
            m.fit(xtr,a.grade); rows.append({'scenario':scenario,'model':name,**metrics(b.grade,m.predict(xte))})
    pd.DataFrame(rows).to_csv(T/'historical_reproduction.csv',index=False)
    return rows

def run():
    
    plt.style.use('default')
    for p in [T,F,OUT/'models']: p.mkdir(parents=True,exist_ok=True)
    before=original_hashes(); write_json(OUT/'original_manifest.json',before)
    df=load_data(); train,test=load_split(df)
    audit={'raw_rows':len(df),'train_rows':len(train),'test_rows':len(test),'classes':df.grade.value_counts().to_dict(), 'missing_salary':int(df.is_salary_missing.sum()),'database':validate_database(df),'split_valid':True}
    write_json(OUT/'data_audit.json',audit)
    print('Validated dataset and split',audit,flush=True)
    reproduce(train,test)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    specs={
      'Dummy':(DummyClassifier(strategy='most_frequent'),{},False),
      'Logistic Regression':(LogisticRegression(max_iter=1000,random_state=SEED),{},False),
      'KNN':(KNeighborsClassifier(n_neighbors=7,weights='distance'),{},False),
      'Random Forest':(RandomForestClassifier(random_state=SEED,n_jobs=1),{'model__n_estimators':[100,200],'model__max_depth':[5,None],'model__min_samples_leaf':[1,5],'model__max_features':['sqrt',0.8]},False),
      'Boosting':(HistGradientBoostingClassifier(random_state=SEED,early_stopping=False),{'model__max_iter':[100,200],'model__max_leaf_nodes':[7,15],'model__l2_regularization':[0,5],'model__learning_rate':[0.05,0.1]},False),
      'MLP':(MLPClassifier(random_state=SEED,early_stopping=False),[
        {'model__hidden_layer_sizes':[(32,)],'model__alpha':[0.001,0.1],'model__learning_rate_init':[0.001],'model__activation':['relu'],'model__max_iter':[250]},
        {'model__hidden_layer_sizes':[(64,32)],'model__alpha':[0.001,0.1],'model__learning_rate_init':[0.003],'model__activation':['relu'],'model__max_iter':[400]},
        {'model__hidden_layer_sizes':[(32,)],'model__alpha':[0.001,0.1],'model__learning_rate_init':[0.003],'model__activation':['tanh'],'model__max_iter':[400]}],True)}
    write_json(OUT/'search_spaces.json',{k:v[1] for k,v in specs.items()})
    models={}; rows=[]; selections={}
    
    for name,(est,grid,scale) in specs.items():
        print('Training CV:',name,flush=True)
        pipe=pipeline(est,scale_all=scale)
        if grid:
            search=GridSearchCV(pipe,grid,scoring='f1_macro',cv=cv,n_jobs=1,return_train_score=True,error_score='raise')
            search.fit(train,train.grade)
            pd.DataFrame(search.cv_results_).to_csv(T/f'{name.replace(" ","_")}_search.csv',index=False)
            model=search.best_estimator_; idx=search.best_index_
            mean=search.best_score_; std=search.cv_results_['std_test_score'][idx]
            selections[name]={'params':search.best_params_,'cv_mean':mean,'cv_std':std}
        else:
            scores=cross_validate(pipe,train,train.grade,cv=cv,scoring='f1_macro',return_train_score=True)
            mean=scores['test_score'].mean(); std=scores['test_score'].std(); model=pipe.fit(train,train.grade)
            selections[name]={'params':{},'cv_mean':mean,'cv_std':std}
            pd.DataFrame(scores).to_csv(T/f'{name.replace(" ","_")}_cv.csv',index=False)
        models[name]=model
        rows.append({'model':name,'cv_mean':mean,'cv_std':std})
        print(name,selections[name],flush=True)
    preferred=max(selections,key=lambda k:selections[k]['cv_mean'])
    write_json(OUT/'selection.json',{'criterion':'Highest mean training 5-fold macro F1; test scores not used','preferred_model':preferred,'models':selections})
    
    pp=Pipeline([('prep',preprocess(scale_all=True)),('pca',PCA(n_components=0.95,svd_solver='full')),('model',LogisticRegression(max_iter=1000,random_state=SEED))])
    ps=cross_validate(pp,train,train.grade,cv=cv,scoring='f1_macro',return_train_score=True)
    pd.DataFrame(ps).to_csv(T/'pca_lr_cv.csv',index=False)
    learning_progress(models['MLP'],train)
    predictions=test[['id','name','grade','salary_mean','is_salary_missing','num_skills','schedule','key_skills','desc_length']].copy()
    classrows=[]; fig,axes=plt.subplots(2,3,figsize=(15,9))
    for row,ax in zip(rows,axes.flat):
        name=row['model']; m=models[name]; pred=m.predict(test)
        row.update(metrics(test.grade,pred)); predictions[name]=pred
        report=classification_report(test.grade,pred,labels=LABELS,output_dict=True,zero_division=0)
        write_json(T/f'{name.replace(" ","_")}_classification_report.json',report)
        classrows.append({'model':name,**{c:report[c]['f1-score'] for c in LABELS}})
        ConfusionMatrixDisplay(confusion_matrix(test.grade,pred,labels=LABELS),display_labels=LABELS).plot(ax=ax,colorbar=False,cmap='Blues'); ax.set_title(name)
        joblib.dump(m,OUT/'models'/f'{name.replace(" ","_")}.joblib')
    savefig('confusion_matrices')
    comparison=pd.DataFrame(rows).sort_values('cv_mean',ascending=False)
    comparison.to_csv(T/'model_comparison.csv',index=False)
    predictions.to_csv(T/'test_predictions.csv',index=False)
    error_rows=[]
    for name in models:
        e=predictions[predictions[name]!=predictions.grade].copy(); e['model']=name; e['predicted']=e[name]
        error_rows.append(e[['id','name','grade','predicted','model','salary_mean','is_salary_missing','num_skills','schedule','key_skills','desc_length']])
    pd.concat(error_rows).to_csv(T/'misclassifications.csv',index=False)
    groups=[]
    for name in models:
        for missing,sub in predictions.groupby('is_salary_missing'):
            groups.append({'model':name,'missing_salary':missing,'n':len(sub),'errors':int((sub.grade!=sub[name]).sum()),'macro_f1':f1_score(sub.grade,sub[name],labels=LABELS,average='macro',zero_division=0)})
    pd.DataFrame(groups).to_csv(T/'error_by_salary.csv',index=False)
    pd.DataFrame(classrows).set_index('model').plot.bar(figsize=(11,5),ylim=(0,1),ylabel='Test class F1',rot=15); savefig('class_f1')
    x=np.arange(len(comparison)); plt.figure(figsize=(10,5)); plt.bar(x-.18,comparison.cv_mean,.36,yerr=comparison.cv_std,label='Training CV mean ± SD'); plt.bar(x+.18,comparison.test_macro_f1,.36,label='Held-out test'); plt.xticks(x,comparison.model,rotation=15); plt.ylim(0,1); plt.ylabel('Macro F1'); plt.legend(); savefig('model_comparison')
    
    ar=[]
    for name,est in [('Logistic Regression',LogisticRegression(max_iter=1000,random_state=SEED)),('KNN',KNeighborsClassifier(n_neighbors=5,weights='distance'))]:
        m=pipeline(est,'A').fit(train,train.grade); ar.append({'scenario':'A','model':name,**metrics(test.grade,m.predict(test))})
    pd.DataFrame(ar).to_csv(T/'scenario_A_corrected.csv',index=False)
    pca_analysis(train,test,predictions,preferred,ps)
    write_json(OUT/'environment.json',{'python':sys.version,'platform':platform.platform(),'versions':{p:__import__(p).__version__ for p in ['numpy','pandas','sklearn','matplotlib','joblib']}})
    assert before==original_hashes(),'Original files changed!'
    write_json(OUT/'integrity_verified.json',{'original_files_unchanged':True,'files_checked':len(before)})
    print(comparison.to_string(index=False),flush=True)
    return comparison

def learning_progress(selected,train):
    
    a,b=train_test_split(train,test_size=.2,stratify=train.grade,random_state=SEED)
    prep=clone(selected.named_steps['prep']); xa=prep.fit_transform(a); xb=prep.transform(b)
    m=clone(selected.named_steps['model']); m.set_params(max_iter=1,early_stopping=False)
    records=[]
    for epoch in range(1,201):
        m.partial_fit(xa,a.grade,classes=np.array(LABELS))
        if epoch==1 or epoch%5==0:
            records.append({'epoch':epoch,'loss':m.loss_,'train_f1':f1_score(a.grade,m.predict(xa),average='macro'),'validation_f1':f1_score(b.grade,m.predict(xb),average='macro')})
    progress=pd.DataFrame(records); progress.to_csv(T/'mlp_learning_progress.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); axes[0].plot(progress.epoch,progress.loss); axes[0].set(xlabel='Epoch',ylabel='Training loss')
    axes[1].plot(progress.epoch,progress.train_f1,label='Train 800'); axes[1].plot(progress.epoch,progress.validation_f1,label='Validation 200'); axes[1].set(xlabel='Epoch',ylabel='Macro F1',ylim=(0,1)); axes[1].legend(); savefig('mlp_learning')
    fit=selected.named_steps['model']; write_json(OUT/'mlp_convergence.json',{'selected_iterations':fit.n_iter_,'max_iter':fit.max_iter,'hit_iteration_cap':fit.n_iter_>=fit.max_iter,'training_loss':fit.loss_,'diagnostic_last':records[-1], 'diagnostic_best_validation':max(r['validation_f1'] for r in records),'note':'Diagnostic on 800/200 training-only holdout; not a test-guided stopping rule'})

def pca_analysis(train,test,predictions,preferred,ps):
    prep=preprocess(scale_all=True); x=prep.fit_transform(train); xt=prep.transform(test)
    pca=PCA(svd_solver='full').fit(x); z=pca.transform(x); zt=pca.transform(xt)
    variance=pca.explained_variance_ratio_; pd.DataFrame({'component':np.arange(1,len(variance)+1),'variance':variance,'cumulative':variance.cumsum()}).to_csv(T/'pca_variance.csv',index=False)
    names=prep.named_steps['columns'].get_feature_names_out()
    pd.DataFrame(pca.components_.T,index=names,columns=[f'PC{i+1}' for i in range(len(variance))]).to_csv(T/'pca_loadings.csv')
    plt.figure(figsize=(8,4)); plt.plot(np.arange(1,len(variance)+1),variance.cumsum(),marker='o'); plt.axhline(.95,color='gray',ls='--'); plt.xlabel('Components'); plt.ylabel('Cumulative explained variance'); savefig('pca_variance')
    plt.figure(figsize=(8,5))
    for c in LABELS:
        mask=train.grade==c; plt.scatter(z[mask,0],z[mask,1],label=c,alpha=.45,s=15)
    wrong=predictions[preferred]!=test.grade
    plt.scatter(zt[wrong,0],zt[wrong,1],marker='x',c='black',s=35,label=f'{preferred} test errors')
    plt.xlabel(f'PC1 ({variance[0]:.1%})'); plt.ylabel(f'PC2 ({variance[1]:.1%})'); plt.legend(fontsize=8); savefig('pca_projection')
    examples=train[['id','name','grade','salary_mean','num_skills','desc_length','key_skills']].copy(); examples['PC1']=z[:,0]; examples['PC2']=z[:,1]
    pd.concat([examples.nsmallest(3,'PC1'),examples.nlargest(3,'PC1'),examples.nsmallest(3,'PC2'),examples.nlargest(3,'PC2')]).drop_duplicates('id').to_csv(T/'pca_examples.csv',index=False)
    write_json(OUT/'pca_summary.json',{'first_two_variance':float(variance[:2].sum()),'components_95':int(np.searchsorted(variance.cumsum(),.95)+1),'lr_pca_cv_mean':float(ps['test_score'].mean()),'lr_pca_cv_std':float(ps['test_score'].std()),'fit_rows':len(train),'features':len(names)})

if __name__=='__main__':
    with threadpool_limits(limits=1): run()


import json
import numpy as np
import pandas as pd
from pptx import Presentation
from pypdf import PdfReader
from src.project import ROOT, metrics, original_hashes

def verify():
    t=ROOT/'results/tables'; comp=pd.read_csv(t/'model_comparison.csv').set_index('model'); predictions=pd.read_csv(t/'test_predictions.csv',dtype={'id':str})
    errors=pd.read_csv(t/'misclassifications.csv',dtype={'id':str})
    for name,row in comp.iterrows():
        for k,v in metrics(predictions.grade,predictions[name]).items():
            assert abs(row[k]-v)<1e-12,(name,k)
        bad=predictions[predictions.grade!=predictions[name]]
        assert set(errors.loc[errors.model==name,'id'])==set(bad.id)
        modelpath=ROOT/'results/models'/f'{name.replace(" ","_")}.joblib'
        assert modelpath.is_file()
    selected=json.loads((ROOT/'results/selection.json').read_text())
    assert selected['preferred_model']==comp.cv_mean.idxmax()
    prs=Presentation(ROOT/'presentation/Endterm_IT_Vacancy_Seniority.pptx')
    assert len(prs.slides)==8
    text=['\n'.join(s.text for s in slide.shapes if s.has_text_frame) for slide in prs.slides]
    for slide in prs.slides:
        assert len(slide.notes_slide.notes_text_frame.text)>100
        for shape in slide.shapes:
            assert shape.left>=0 and shape.top>=0
            assert shape.left+shape.width<=prs.slide_width+1000
            assert shape.top+shape.height<=prs.slide_height+1000
    for model,idx in [('Logistic Regression',1),('KNN',1),('Random Forest',3),('Boosting',3),('MLP',5)]:
        assert f'{comp.loc[model,"test_macro_f1"]:.4f}'[1:] in text[idx],(model,'slide score mismatch')
    pdf=PdfReader(ROOT/'presentation/Endterm_IT_Vacancy_Seniority.pdf'); assert len(pdf.pages)==8
    for i,page in enumerate(pdf.pages):
        assert text[i].split('\n')[0] in page.extract_text(),('PDF title mismatch',i)
    nb=json.loads((ROOT/'notebooks/Endterm_IT_Vacancy_Seniority.ipynb').read_text())
    code=[c for c in nb['cells'] if c['cell_type']=='code']; assert len(code)>=3
    assert all(c['execution_count'] is not None for c in code)
    assert not any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[]))
    assert original_hashes()==json.loads((ROOT/'results/original_manifest.json').read_text())
    evidence=json.loads((ROOT/'presentation/slide_evidence.json').read_text()); assert sum(s['seconds'] for s in evidence)==420
    clean=json.loads((ROOT/'results/clean_environment_verified.json').read_text()); assert clean['metrics_match'] and clean['tests_passed']
    (ROOT/'results/artifact_audit.json').write_text(json.dumps({'model_count':len(comp),'prediction_rows':len(predictions),'all_metrics_recomputed':True,'errors_verified':True,'slides':8,'pdf_pages':8,'notebook_executed_without_errors':True,'slide_metrics_verified':True,'original_files_unchanged':True,'clean_environment_verified':True,'speaking_seconds':420},indent=2))
    print('Artifact audit passed: metrics, errors, slides/PDF, executed notebook, integrity and clean environment.')
if __name__=='__main__': verify()

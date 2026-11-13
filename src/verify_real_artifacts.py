"""Independent consistency checks for the real-data submission artifacts."""
import json, hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.real_experiments import ROOT, OUT, LABELS, NUM, CAT

def verify():
    audit=json.loads((OUT/'data_audit.json').read_text())
    assert hashlib.sha256((ROOT/'data/real/djinni_jobs.parquet').read_bytes()).hexdigest()==audit['sha256']
    df=pd.read_csv(ROOT/'data/real/benchmark.csv').set_index('id',drop=False)
    a=pd.read_csv(ROOT/'data/real/train_ids.csv');b=pd.read_csv(ROOT/'data/real/test_ids.csv')
    assert not set(a.id)&set(b.id)
    assert set(a.id)|set(b.id)==set(df.index)
    assert not set(a.company_group)&set(b.company_group)
    test=df.loc[b.id]
    pred=pd.read_csv(OUT/'tables/test_predictions.csv').set_index('id')
    assert pred.index.tolist()==test.index.tolist()
    assert pred.target.tolist()==test.target.tolist()
    comp=pd.read_csv(OUT/'tables/model_comparison.csv')
    for r in comp.itertuples():
        assert np.isclose(f1_score(pred.target,pred[r.model],average='macro'),r.test_macro_f1)
        assert np.isclose(accuracy_score(pred.target,pred[r.model]),r.test_accuracy)
        assert (pred.target!=pred[r.model]).sum()==r.errors
        cm=pd.read_csv(OUT/'tables'/f'{r.model.replace(" ","_")}_confusion.csv',index_col=0)
        np.testing.assert_array_equal(cm,confusion_matrix(pred.target,pred[r.model],labels=LABELS))
        model=joblib.load(OUT/'models'/f'{r.model.replace(" ","_")}.joblib')
        np.testing.assert_array_equal(model.predict(test),pred[r.model])
        # A frame containing ONLY approved features is sufficient to reproduce predictions.
        np.testing.assert_array_equal(model.predict(test[NUM+CAT]),pred[r.model])
    result={'all_six_saved_models_match_predictions':True,'metric_recalculation_matches':True,'confusion_matrices_match':True,'source_hash_matches':True,'split_ids_and_order_verified':True,'company_overlap':0,'only_approved_inputs_required':True}
    (OUT/'artifact_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':verify()

# IT Vacancy Seniority Classification — Endterm
Machine Learning Algorithms, Astana IT University  
Students: Zholdubayev Diyar and Galymzhan Turemuratov.

This is a synthetic-data experiment, continuing the immutable Midterm snapshot. Labels are derived from title and experience, not independently verified seniority. The CSV has 1,250 rows; the saved split has 1,000 train and 250 test IDs. All raw fields match the SQLite database. Scenario B excludes title/name, experience and experience_id and is the primary task. Scenario A is a separate shortcut-learning demonstration.

## Verified results
| model | cv_mean | cv_std | test_accuracy | test_macro_f1 | test_precision | test_recall |
| --- | --- | --- | --- | --- | --- | --- |
| Boosting | 0.8695 | 0.0119 | 0.8480 | 0.8478 | 0.8514 | 0.8472 |
| Random Forest | 0.8521 | 0.0152 | 0.8400 | 0.8403 | 0.8394 | 0.8433 |
| Logistic Regression | 0.8451 | 0.0228 | 0.8160 | 0.8172 | 0.8166 | 0.8205 |
| MLP | 0.8048 | 0.0207 | 0.8080 | 0.8093 | 0.8094 | 0.8104 |
| KNN | 0.7934 | 0.0190 | 0.8000 | 0.7988 | 0.8072 | 0.8028 |
| Dummy | 0.1822 | 0.0007 | 0.3760 | 0.1822 | 0.1253 | 0.3333 |

Macro F1 is primary because it weights Junior, Middle and Senior equally and penalizes neglect of a class. Precision/recall are macro averages; CV spread is population standard deviation across five fixed stratified folds, not a confidence interval. Boosting is preferred by CV, before test predictions. Search CV scores have selection optimism; no nested-CV claim is made. The test set was already seen during Midterm and cannot be treated as a new external evaluation.

## Run from the repository root
Use Python 3.15 (verified here; Python 3.11+ may work but was not tested).

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.run_experiments
python -m pytest -q
python -m src.build_deliverables
python -m src.package_submission --group YOUR_GROUP
```

Open notebooks/Endterm_IT_Vacancy_Seniority.ipynb and restart/run all using this environment. The notebook imports and calls the same experiment functions directly; no collection or split generation occurs. The run performs 5-fold GridSearchCV for RF (16 settings), boosting (16) and MLP (6), plus fixed baselines. It takes a few minutes on a laptop. All model choices are fixed in src/run_experiments.py. Joblib files are for trusted local use and require the matching dependency versions. requirements-lock.txt captures the actual environment.

## Submission map
- data/raw/: unchanged original CSV, SQLite and saved IDs. Historical baseline reference is src/_midterm_reference.py; other Midterm sources are in the local ignored archive/midterm_originals.zip.
- data/: original SHA-256 inventory and pointer to canonical data.
- src/: loading, validation, fold-safe pipelines, experiments, documentation/deck and packaging.
- notebooks/: executed Endterm notebook; original Midterm notebook is archived locally and is not part of the Git working tree.
- results/tables/: CV searches, consolidated metrics, reports, predictions and actual errors.
- results/figures/: confusion matrices, class F1, PCA, learning progress and model comparison.
- results/models/: six fitted Scenario B pipelines.
- presentation/: eight-slide PPTX, talk script, slide evidence mapping and previews if available.
- CHANGELOG.md, CONCLUSIONS.md, DEFENSE_QA.md, CONTRIBUTIONS.md, RUBRIC_CHECKLIST.md.

## Important limits
The generator controls salary and skill distributions; description templates can include the title. Excluding direct columns does not remove every synthetic shortcut. Original skill matching uses substrings (Go can match Django) and is retained for exact continuity, explicitly disclosed. Currency is RUR throughout; no cross-currency salary comparison is attempted. None of these results establishes real-market validity.

No Git commit or push is performed. The archive uses group SE2418, inferred from the existing Midterm submission filename; confirm it before Moodle upload. Both partners must upload the same complete archive. An eight-page PDF was exported with Microsoft PowerPoint; see results/verification.md for verification details.

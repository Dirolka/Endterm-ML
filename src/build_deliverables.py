
from pathlib import Path
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from src.project import ROOT, LABELS
T=ROOT/'results/tables'
def readj(p): return json.loads((ROOT/p).read_text())
def table(df):
    cols=list(df.columns); lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in df.itertuples(index=False,name=None): lines.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |')
    return '\n'.join(lines)
def write(name,text): (ROOT/name).write_text(text.strip()+'\n')

def build():
    comp=pd.read_csv(T/'model_comparison.csv'); sel=readj('results/selection.json'); pca=readj('results/pca_summary.json'); mlp=readj('results/mlp_convergence.json')
    errors=pd.read_csv(T/'misclassifications.csv'); groups=pd.read_csv(T/'error_by_salary.csv'); preds=pd.read_csv(T/'test_predictions.csv')
    report=readj('results/tables/Boosting_classification_report.json')
    write('README.md',f'''# IT Vacancy Seniority Classification — Endterm
Machine Learning Algorithms, Astana IT University  
Students: Zholdubayev Diyar and Galymzhan Turemuratov.

This is a synthetic-data experiment, continuing the immutable Midterm snapshot. Labels are derived from title and experience, not independently verified seniority. The CSV has 1,250 rows; the saved split has 1,000 train and 250 test IDs. All raw fields match the SQLite database. Scenario B excludes title/name, experience and experience_id and is the primary task. Scenario A is a separate shortcut-learning demonstration.

## Verified results
{table(comp)}

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
''')
    hist=pd.read_csv(T/'historical_reproduction.csv'); corr=comp[comp.model.isin(['Logistic Regression','KNN'])]
    write('CHANGELOG.md',f'''# Midterm continuity and corrections
No written instructor feedback was supplied; fixes address observed weaknesses and the rubric.

| Original weakness | Exact correction | Reason | Evidence / remaining limitation |
| --- | --- | --- | --- |
| README/notebook claim API collection and independently meaningful ground truth | Endterm identifies seeded synthetic vacancies and heuristic labels | Accurate provenance | Generator, raw snapshot, data audit; real-market validity remains unknown |
| Global median salary imputation before split | SimpleImputer inside ColumnTransformer and Pipeline, fitted separately in each CV training fold | Prevent validation/test distribution leakage | tests/test_project.py tests fold medians and invariance to validation perturbation |
| Baseline regenerates/writes train/test IDs | Load saved ID CSVs in saved order; reject duplicate, overlapping, unknown or missing IDs | Preserve exact continuity | results/data_audit.json and original checksum verification |
| Schedule dummy vocabulary learned globally | OneHotEncoder fitted within folds; handle_unknown=ignore | Avoid transductive schema learning | Unseen category test |
| Historical values had no independent prediction logs | Reconstructed original preprocessing without calling its writing main; logged predictions for corrected models | Verify values honestly | Tables below; tiny differences below four decimals are rounding |
| No tuned ensembles or MLP learning evidence | Five-fold training-only grids and explicit 800/200 learning diagnostic | Fair model selection and convergence analysis | Search tables, learning curve; several MLP fits hit cap |
| Midterm narrative error claims not backed by records | Saved all test predictions and per-model errors with raw features | Traceable error analysis | results/tables/misclassifications.csv; synthetic examples only |

## Before/after evidence
Historical reported A: LR accuracy .9920 / macro F1 .9918; KNN .8840 / .8848.
Historical reported B: LR .8160 / .8172; KNN .8000 / .7988.
Independently reconstructed original (including global imputation):

{table(hist)}

Corrected Scenario B:

{table(corr)}

Corrected Scenario A matches reproduced A at four decimals (results/tables/scenario_A_corrected.csv).
The imputation fix does not change rounded held-out metrics on this split. That does not make the old method valid: the correction establishes proper fold boundaries, independently tested. Historical and corrected preprocessing differ in where medians and category vocabularies are learned, not in saved rows or target rule.

## Source inconsistencies inspected
- Original README and notebook claim October 2026 API collection/verified terms. No collection logs support that claim; supplied generator and provenance establish synthetic origin.
- Available presentation is IT Vacancy Seniority.pdf; requested IT Vacancy Seniority(1).pdf is absent. This presentation already corrects provenance, acknowledges imputation leakage and calls metrics unverified.
- build_notebook.py and notebook retain [Partner Name]; README has both students. Contributions are historical assertions, not newly verified authorship.
- Generator docstring says 1,200 rows while default generates 1,250. Its comment says 10% title replacement but code uses 15%.
- Generator calls the raw snapshot authentic, which is misleading; generated IDs are not API identity evidence.
- Collector is an available acquisition script, not evidence it produced this snapshot. Its TLS verification is disabled; Endterm does not invoke it.
- Original script uses globally imputed salary; notebook builder follows the same pattern. Original script rewrites saved splits and reported metrics when executed; Endterm bypasses those actions.
- Original substring skill flags can match Go inside Django. Preserved to reproduce baselines; exact normalized matching should be a future training-CV ablation.
- Description template can embed title, so length/word counts retain indirect target cues. Scenario B removes direct proxies, not all synthetic correlations.
- Original heuristic checks noExperience before Senior markers; therefore a Senior/Lead title can become Junior. The same rule is preserved, not silently relabeled.

## Remaining limitations
Synthetic generator patterns, repeated templates, heuristic inconsistencies, previously inspected test set, no nested-CV estimate, small subgroup counts, MLP nonconvergence, single seed/split, and unverified human contributions are retained and disclosed. Original files are unchanged, including historically inaccurate claims, as archival evidence.
''')
    example=errors[errors.model=='Boosting'].head(5)
    pexamples=pd.read_csv(T/'pca_examples.csv')
    loadings=pd.read_csv(T/'pca_loadings.csv',index_col=0)
    permodel=[]
    for model in comp.model:
        r=readj('results/tables/'+model.replace(' ','_')+'_classification_report.json')
        worst=min(LABELS,key=lambda c:r[c]['f1-score'])
        e=errors[errors.model==model]; pairs=e.groupby(['grade','predicted']).size().sort_values(ascending=False)
        permodel.append(f"- **{model}**: {len(e)} errors; weakest F1 class {worst} ({r[worst]['f1-score']:.4f}); most common error {pairs.index[0][0]} → {pairs.index[0][1]} ({pairs.iloc[0]} rows).")
    write('CONCLUSIONS.md',f'''# Findings and Final project plan
## Classification question
Salary, skill flags, schedule and description statistics predict the heuristic labels within this synthetic snapshot. Boosting was chosen using training CV macro F1 {comp.iloc[0].cv_mean:.4f} ± {comp.iloc[0].cv_std:.4f}; its held-out test macro F1 is .8478 and accuracy .8480. LR is the strongest corrected Midterm baseline by training CV (.8451), with test macro F1 .8172. Boosting improves test macro F1 by .0306 and RF by .0230. These are descriptive differences, not significance claims; CV spreads overlap for some models and the test has only 250 observations.

Both ensembles improve over LR on this split; boosting's small trees learn nonlinear thresholds/interactions and RF averages diverse trees. Selected RF uses 100 trees, unrestricted depth, leaf size 1 and sqrt feature sampling. Boosting uses 100 iterations, 7 leaves, rate .05 and L2=0. RF grid covers depth/leaf regularization and tree count; boosting covers leaf complexity, shrinkage, L2 and boosting rounds. We deliberately bound compute, and do not claim global-optimal settings. Full settings and candidates are in results/search_spaces.json and search CSVs.

The MLP does not improve over LR: test F1 .8093 and CV .8048. Selected architecture is 20 inputs → 32 ReLU units → 3-class softmax, Adam, alpha=.1, learning rate=.001, max_iter=250. All input features are standardized within each fold. Six configurations cover relu/tanh, one/two hidden layers, two alpha values, two learning rates and 250/400 iteration limits, bundled as feasible configurations rather than a full factorial search. The selected final model reaches 250 iterations, so numerical convergence is not established. At epoch 200 of the separate training-only diagnostic, train F1 is {mlp['diagnostic_last']['train_f1']:.4f}, validation F1 {mlp['diagnostic_last']['validation_f1']:.4f}; best recorded validation F1 is {mlp['diagnostic_best_validation']:.4f}. Falling loss plus a train/validation gap supports some overfitting and weaker generalization. There is no basis to claim pure underfitting or full convergence. The curve is a fixed-setting diagnostic, not test-guided stopping, and its model is distinct from the final full-training fit. No test-driven tuning follows these observations.

## PCA evidence
PCA is fit only on 1,000 training rows after median imputation and standardization of all 20 encoded features, including binary columns. This avoids salary units dominating covariance; scaling rare binary features can also amplify them. First two components retain {pca['first_two_variance']:.2%} variance, and {pca['components_95']} components reach 95%. The 2D map discards most information and cannot establish real seniority groups.
PC1 has positive salary (.4375), description length (.4020), word count (.3936), and negative Git (-.3054) loadings. PC2 primarily reflects description words (.5003), length (.4899), and negative salary (-.3192). Signs are arbitrary; relative contributions matter. Repeated templates can dominate these directions.

Actual training examples at projection extremes:
{table(pexamples[['id','grade','salary_mean','num_skills','desc_length','PC1','PC2']].head(8))}

ID 90000480 is a low-salary Junior near negative PC1. ID 90000654 has a Lead/Senior title and salary 530000 but heuristic Junior label, showing why projection overlap can reflect label-rule conflicts rather than a model failure to discover true classes. The full example table includes skill lists and titles. Black crosses on the projection locate preferred-model test errors without refitting PCA on test rows.
A fold-fitted 95%-variance PCA+LR diagnostic has CV F1 {pca['lr_pca_cv_mean']:.4f} ± {pca['lr_pca_cv_std']:.4f}, below original corrected LR .8451 ± .0228. PCA is therefore explanatory here, not adopted for final classification. This diagnostic also changes scaling of binary columns, so the difference cannot be attributed solely to dimensionality reduction. It is not included in the six-model primary ranking. Variance retention, rather than untested cluster stability, is the relevant diagnostic; no clustering claims are made.

## Errors across all models
{chr(10).join(permodel)}

Boosting class F1 is Junior .8780, Middle .8377 and Senior .8276. Senior recall .7792 is lowest; the hardest class depends on model, so a generic claim that Middle is always hardest would be inaccurate. The class F1 chart and six confusion matrices show the differences.

Actual misclassified synthetic records (titles shown for inspection only, not fed to Scenario B):
{table(example[['id','name','grade','predicted','salary_mean','num_skills']])}

Salary missingness breakdown:
{table(groups)}

Boosting makes 19 errors among 60 rows without salary (31.7%) and 19 among 190 with salary (10%). Salary absence weakens an important generator signal; skills and label conflicts then matter more. These observational subgroups do not establish causality. KNN suffers especially without salary; an imputed midpoint makes unlike records appear close. LR is compact and interpretable, RF captures interactions without scaling sensitivity, boosting adds sequential corrections, and MLP requires scaling and more tuning/compute without improvement here. Dummy predicts Middle only and exposes the weakness of accuracy alone. Error records are exhaustive, not hand-created anecdotes.

## Limitations and next step
These findings apply to seeded synthetic records and heuristic targets. Generated salary bands and skill priorities build seniority-like separability into data. Identical description templates, heuristic override ordering, Go/Django substring matching, and title-derived lengths complicate interpretation. Models can reproduce generator structure without understanding hiring. Previously exposed Midterm test data, a single fixed split/seed and selected-search CV optimism limit confidence. New real data must be evaluated separately; do not replace this Endterm dataset or relabel it retrospectively.

For the Final, use the existing raw HTML description field, already attached to each synthetic record by its immutable ID. This is a lawful local synthetic source and requires no website access. A synthetic ID cannot lawfully or scientifically be matched to a real hh.ru vacancy ID; no claim of real identity is allowed. Clean HTML, mask explicit seniority words and copied title spans, and build TF-IDF word/character features with vocabulary/IDF fitted inside training folds. Compare tabular-only, text-only and combined models on the exact same saved train/test IDs using macro F1, class reports, missing-salary subgroups and real error records. Select representations and regularization by training CV; publish gains or failures without test-guided retuning. Template-grouped or near-duplicate-grouped training CV is a sensitivity check against memorization. Report masked vs unmasked training-CV ablations to reveal target leakage.

A later external-validity study could collect descriptions through the official HeadHunter API only after checking the current API terms, permissions, retention rules and rate limits; use documented User-Agent, TLS verification, minimal fields, no personal contact data, and record timestamp/license/access conditions. Match descriptions to the same newly collected vacancies by verified API ID, never by guessed title or synthetic ID. Independently double-label a consented/permitted sample with an annotation rubric and agreement checks; preserve employer/time grouping and a genuinely untouched external test. Treat this as a separate benchmark, not a replacement for the preserved Endterm. We have not verified current API access rights and make no claim they permit a proposed collection today.
''')
    write('CONTRIBUTIONS.md','''# Contributions and AI disclosure
## Verified activity in this Endterm session
Zholdubayev Diyar requested and directed this continuation in the repository. OpenAI Codex performed the code implementation, execution, debugging, metric analysis, tests, notebook construction, documentation, presentation generation and packaging in this session. These automated activities must not be presented as independently handwritten student code. Diyar's human review, understanding and rehearsal remain to be completed and verified; no claim of manual verification is made on his behalf.

Galymzhan Turemuratov's new Endterm implementation/review contribution is **unassigned and unverified**. Do not credit new code to him until evidence is supplied. The proposed speaking allocation is a rehearsal plan, not proof of authorship or completed defense participation.

## Historical attribution
The original README attributes collector/preprocessing to Galymzhan and baselines/presentation to Diyar. This is retained as a historical declaration, not independently verified authorship. The notebook still contains a partner placeholder. Historical Gemini/Antigravity and GitHub Copilot use is reported by the original README, not newly verified.

## AI tools actually used now
OpenAI Codex assisted with inspection, Python/scikit-learn implementation, training searches, error analysis, automated tests, notebook execution, writing, python-pptx slides, defense Q&A and packaging. PyCharm IDE tools supplied interpreter information and notebook execution. Scientific results came from executed local code, not language-model estimates. Both students must review every part, correct any errors, and explain it independently. No Git commit or push was performed.

## Before submission
Record each student's verified review/rehearsal work and the correct course group. Both students should run the project and practice all questions. Do not retroactively assign automated implementation to a partner.
''')
    qa(comp,sel,pca,mlp)
    deck(comp,sel,pca,mlp)
    write('RUBRIC_CHECKLIST.md','''# Evidence audit against the eight rubric criteria
This is an evidence audit, not a guaranteed grade. Criteria 1–7 are team points (85); criterion 8 is individual (15).

| Criterion | Excellent requirement | Evidence / location | Verification status | Remaining issues |
| --- | --- | --- | --- | --- |
| 1 Continuity / fixes (10) | Same data and split; fixes with before/after; reproducible outputs | src/project.py; CHANGELOG.md; results/original_manifest.json; historical_reproduction.csv; notebooks/Endterm_IT_Vacancy_Seniority.ipynb | Data and split audited; historical values independently reproduced; corrected pipeline executed | Original notebook intentionally archival; clean-environment status in verification.md; no instructor feedback supplied |
| 2 Ensembles (15) | RF and boosting tuned in training CV with fold preprocessing, spaces and spread | src/run_experiments.py; results/search_spaces.json; selection.json; RF/Boosting search tables | Both 16-setting grids completed with 5 folds, saved split used | Search is bounded; selected CV is optimistic; no significance claim |
| 3 Unsupervised (10) | Scaled PCA interpreted with real objects, variance and baseline connection | pca_variance/projection figures; pca_loadings/examples tables; pca_lr_cv.csv; CONCLUSIONS.md | Training-fitted PCA and fold-fitted PCA+LR diagnostic executed | 2D preserves only 20.95%; diagnostic also changes binary scaling; no true-group claim |
| 4 MLP (15) | Architecture, train-only learning progress, fair comparison, convergence interpretation | MLP_search.csv; mlp_learning_progress.csv; mlp_convergence.json; mlp_learning.png | Six configurations × five folds; explicit 800/200 curve; same test | Selected fit hits 250 cap; improvement absent; convergence not claimed |
| 5 Fair comparison / errors (15) | All six models, CV spread, same test, justified metric, actual errors | model_comparison.csv; test_predictions.csv; misclassifications.csv; class reports; confusion_matrices.png; CONCLUSIONS.md | Full predictions and metrics saved; preferred model selected before test evaluation | Historical test exposure; small subgroups; no external test |
| 6 Conclusions / Final (10) | Results-based conclusions; lawful source, object matching, feasible evaluation | CONCLUSIONS.md | Actual results interpreted; existing local synthetic descriptions identified; leakage controls and external plan specified | No verified current API permissions; synthetic IDs cannot match real listings |
| 7 Presentation (10) | Clear readable 7-minute talk, labelled figures, both speak | presentation/Endterm_IT_Vacancy_Seniority.pptx; presentation/SCRIPT.md; slide_evidence.json | Eight slides generated from saved metrics; proposed 420-second allocation | Actual delivery/rehearsal and both students speaking require human verification; rendering status in verification.md |
| 8 Individual defense (15 each) | Explain all methods, contributions and AI use independently | DEFENSE_QA.md (48 questions); CONTRIBUTIONS.md | Preparation materials completed and AI disclosed | Cannot verify human understanding or guarantee individual marks; partner Endterm contribution unverified |

Submission filename and any converter/clean-environment limitations are tracked in results/verification.md. The same complete archive should be given to both students.
''')

def qa(comp,sel,pca,mlp):
    pairs=[
('What is the project question?','Can indirect vacancy features predict Junior, Middle or Senior heuristic labels in the preserved synthetic snapshot?'),
('Are these real HeadHunter vacancies?','No. They are 1,250 synthetic records generated with seed 42 in an API-like schema.'),
('What is one observation?','One synthetic vacancy record with a unique synthetic ID and ten original raw fields.'),
('Where do labels come from?','The original assign_grade rule uses title markers and experience. They are heuristic labels, not independently verified truth.'),
('Why can a Senior title become Junior?','The rule checks Junior markers and noExperience before Senior markers. Preserve this conflict for continuity rather than silently changing labels.'),
('How is the saved split preserved?','load_split reads train_ids.csv and test_ids.csv in saved order; it never calls train_test_split for the final held-out partition.'),
('What split checks run?','Unique raw/split IDs, 1000/250 counts, disjoint partitions, existence and exact union coverage. CSV and SQLite raw fields are compared.'),
('Why not regenerate with seed 42?','Saved IDs define the experimental contract. A seed alone is weaker because row order, libraries or target rules could change.'),
('What is Scenario A?','Indirect features plus title TF-IDF and experience encoding. It demonstrates target shortcuts and is excluded from primary ranking.'),
('What is Scenario B?','Salary midpoint/missingness, description length/words, skill count, twelve skill flags and schedule. No name, experience or experience_id inputs.'),
('Is Scenario B completely free of shortcuts?','No. The generator creates correlations and some descriptions contain copied titles, so length statistics can retain indirect target cues.'),
('What was wrong with salary imputation?','The old code computed the median on all rows before splitting, allowing test distribution information into training preprocessing.'),
('How is that fixed?','SimpleImputer is inside the numeric pipeline inside ColumnTransformer. GridSearchCV clones and fits it only on each fold training rows.'),
('Did fixing leakage improve the baseline score?','Rounded test scores did not change: LR F1 .8172 and KNN .7988. Correct methodology matters even when the score is unchanged.'),
('What statistics are learned?','Salary median, numeric means/scales, schedule category vocabulary, title vocabulary/IDF in A, all-feature scaling for MLP/PCA and PCA components.'),
('Which transformations can happen before splitting?','Deterministic per-row HTML cleanup, skill flags from a fixed list, counts and unchanged target rule. They learn no cross-row statistics.'),
('What is a Pipeline?','An estimator that fits preprocessing then the model together. During CV it prevents preprocessing from seeing validation rows.'),
('What is ColumnTransformer?','It applies different transformations to numeric, skill and categorical columns, combines them and drops other columns.'),
('Why standardize for LR, KNN and MLP?','Otherwise large salary units dominate distances and optimization. Baseline numeric scaling is retained; MLP scales every encoded column.'),
('How are unknown schedules handled?','OneHotEncoder(handle_unknown=ignore) returns zeros for the unseen category without learning it from test rows.'),
('Why macro F1?','It averages each class F1 equally. F1 balances precision and recall, so a model cannot hide weak Senior predictions behind the largest class.'),
('What do precision and recall mean?','Precision asks how many predictions of a class are correct. Recall asks how many actual members of that class are found.'),
('What does DummyClassifier show?','Always predicting Middle achieves accuracy .376 but macro F1 .1822, showing that accuracy alone is not sufficient.'),
('How is CV stratified?','Five shuffled folds with seed 42 preserve approximate class proportions, using only the 1000 saved training rows.'),
('What is CV spread?','Population standard deviation of five validation macro F1 scores. It measures fold variation, not a formal confidence interval.'),
('What is random forest?','Many randomized decision trees vote. Averaging reduces variance relative to a single tree.'),
('What RF parameters were searched?','Trees 100/200, depth 5/unlimited, minimum leaf size 1/5 and feature sampling sqrt/.8: sixteen settings.'),
('What RF settings won?','100 trees, unlimited depth, minimum leaf size 1 and sqrt feature sampling. CV macro F1 .8521 ± .0152.'),
('How does boosting differ?','It builds trees sequentially to improve the current ensemble through gradient-based corrections, rather than independent voting trees.'),
('What boosting settings were searched and selected?','Search: 100/200 rounds, 7/15 leaves, L2 0/5 and learning rate .05/.1. Selected: 100 rounds, 7 leaves, L2 0, rate .05.'),
('Which model is preferred and why?','Boosting has the highest mean training CV macro F1, .8695 ± .0119. Test F1 .8478 is final reporting, not the selection rule.'),
('Can you claim the difference is statistically significant?','No. We did not run a prespecified inferential comparison, and selected-search CV means have optimism. Differences are descriptive.'),
('What does PCA do?','It finds orthogonal directions of maximum variance without using labels. These directions need not predict seniority.'),
('Why scale before PCA?','PCA follows variance, so salary units would dominate. We standardize all 20 encoded features using training data only.'),
('What did explained variance show?','The first two PCs retain 20.95% variance. Sixteen components reach 95%, so a two-dimensional picture loses most information.'),
('How are PCA components interpreted?','PC1 combines salary and description size against Git. PC2 mainly reflects description size and inverse salary. Loadings and actual extreme records support this.'),
('Did PCA help LR?','Fold-fitted 95%-variance PCA+LR scored .7905 ± .0231 CV versus .8451 for corrected LR. Binary scaling also changed, so do not attribute the whole difference solely to PCA.'),
('Do PCA clusters prove real categories?','No. Labels were only colors for interpretation; synthetic templates and lost dimensions can create or hide apparent groups.'),
('What MLP configurations were investigated?','Six settings cover (32,) or (64,32), relu/tanh, alpha .001/.1, learning rates .001/.003, and max_iter 250/400.'),
('What is the selected MLP architecture?','Twenty scaled inputs, one hidden layer of 32 ReLU units, three-class softmax output; Adam, alpha .1, learning rate .001, max_iter 250.'),
('What do alpha and learning rate control?','Alpha penalizes large weights to reduce overfitting. Learning rate sets optimizer step size; too large may be unstable, too small may converge slowly.'),
('How was MLP learning checked without leakage?','Split training rows into 800/200 before fitting preprocessing. partial_fit records loss and train/validation macro F1 through 200 epochs.'),
('Did the MLP converge and generalize well?','Selected fit hit 250 iterations, so convergence is not established. Diagnostic train F1 .8996 versus validation .8253 suggests some overfitting; test .8093 is below LR.'),
('Which class and subgroup are difficult?','For boosting Senior has lowest F1 .8276 and recall .7792. Salary-missing rows have 19/60 errors versus 19/190 when salary exists.'),
('How do you inspect actual errors?','misclassifications.csv records ID, true/predicted heuristic label, model, salary, skills and description size. Cross-check it against test_predictions.csv and confusion matrices.'),
('What are the main generalization limits?','Synthetic distributions, repeated templates, label-rule conflicts, substring skill flags, small subgroups, one split and historical test exposure prevent real-market claims.'),
('What is the Final text plan and lawful source?','Use existing local synthetic descriptions joined by unchanged synthetic ID, mask title/seniority clues, and fit TF-IDF within training folds. A separate real API study requires current permission checks and verified real IDs; never match synthetic IDs to live jobs.'),
('Who did the work and what AI was used?','OpenAI Codex implemented and executed this Endterm work under Diyar’s request. Partner new work is unverified. Prior Gemini/Copilot use is only a historical README statement. Both students must review, rehearse and explain the code independently.')]
    assert len(pairs)>=40
    write('DEFENSE_QA.md','# Defense preparation\n\nBoth students should practice every answer. These are evidence-based preparation notes, not proof of understanding.\n\n'+'\n\n'.join(f'### {i}. {q}\n{a}' for i,(q,a) in enumerate(pairs,1)))

def deck(comp,sel,pca,mlp):
    prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
    navy=RGBColor(18,32,54); ink=RGBColor(32,47,67); accent=RGBColor(0,115,136)
    titles=['Project Overview','Dataset and Midterm Continuity','Preprocessing Fixes and Leakage','Tuned Random Forest and Boosting','PCA / Unsupervised Analysis','MLP Neural Network','Model Comparison and Error Analysis','Conclusions and Final Plan']
    copy=[
      ['IT Vacancy Seniority Classification','Junior · Middle · Senior','Machine Learning Algorithms | Astana IT University','Zholdubayev Diyar and Galymzhan Turemuratov','Synthetic-data experiment; heuristic labels'],
      ['1,250 synthetic records; 10 raw fields','Saved split: 1,000 train / 250 test','IDs, coverage and CSV–SQLite equality verified','Reproduced LR F1 .8172; KNN .7988','Original data and sources remain unchanged'],
      ['Before: full-dataset salary median','After: fold-fitted imputer → scaler → classifier','Scenario A: title + experience are target proxies','Primary Scenario B excludes all direct proxies','Rounded baseline test scores are unchanged'],
      ['Five-fold stratified training CV; 16 settings each','RF: 100 trees; depth unlimited; leaf 1; sqrt','CV F1 .8521 ± .0152 | Test .8403','Boosting: 100 rounds; 7 leaves; rate .05; L2 0','CV F1 .8695 ± .0119 | Test .8478'],
      ['Train-fitted scaling + PCA on 20 encoded features','First 2 PCs: 20.95% variance; 16 PCs: 95%','PC1: salary and description size versus Git','PCA+LR CV F1 .7905; corrected LR .8451','2D overlap cannot establish real seniority groups'],
      ['6 configurations × 5 training folds','Selected: 32 ReLU units; alpha .1; rate .001','250-iteration cap reached: convergence uncertain','800/200 diagnostic: train .8996 / val .8253 F1','Test F1 .8093: no improvement over LR'],
      ['Preferred by CV: Boosting; test F1 .8478','Senior: lowest boosting F1 .8276; recall .7792','Without salary: 19/60 errors; with salary: 19/190','ID 90000925: heuristic Senior → predicted Middle'],
      ['Both ensembles improve on corrected LR here','MLP adds complexity without a score gain','Synthetic data cannot establish market validity','Final: existing descriptions joined by synthetic ID','Mask title clues; compare fold-fitted TF-IDF + tabular']]
    figures={4:'pca_projection.png',5:'mlp_learning.png',6:'model_comparison.png'}
    
    seconds=[40,50,55,60,55,55,60,45]
    scripts=[
'''Our project asks whether indirect vacancy features can predict Junior, Middle and Senior labels without seeing title and experience. This is the Endterm continuation of our Midterm experiment. The dataset is synthetic, and the labels are generated by a heuristic rather than independently checked by employers. We therefore study model behavior and leakage, not the accuracy of a real hiring product. We compare six model families on the exact saved split and select the preferred model using training cross-validation.''',
'''The snapshot contains 1,250 records and ten raw fields. We load the saved 1,000 training and 250 test IDs rather than create another split. Automated checks verify unique IDs, complete coverage, no overlap, and equality of every raw field between CSV and SQLite. We preserved all original files. The README and original notebook claimed live API collection, but the generator and revised Midterm presentation identify synthetic data. Independently reconstructing the original preprocessing reproduced Logistic Regression macro F1 of 0.8172 and KNN of 0.7988. These are verified historical values now, not assumed results.''',
'''The main correction moves salary median imputation into the training pipeline. Previously, the median used all rows, including the held-out test data. Now every cross-validation fold learns its own median, scaling and category vocabulary using only training rows. The held-out data only pass through transform. The rounded baseline test scores happen to remain unchanged, but the correction is still necessary. Scenario A includes title and experience, the same variables that create the labels, so it is a shortcut demonstration. Scenario B drops all three direct proxy columns. It still has synthetic correlations, including description length clues, so we avoid calling it proof of real-world generalization.''',
'''We tuned both ensembles with five stratified folds using only the saved training set. Each search evaluates sixteen meaningful settings. Random Forest varies tree count, depth, leaf size and feature sampling. Its selected model uses one hundred trees, unrestricted depth and leaf size one. Its CV macro F1 is 0.8521 and test F1 is 0.8403. Boosting varies rounds, leaf complexity, shrinkage and L2 regularization. Its selected model uses one hundred rounds, seven leaves and learning rate 0.05. It has the highest CV macro F1, 0.8695, so it was selected before test reporting. Its test F1 is 0.8478. These are descriptive improvements, not significance claims.''',
'''PCA uses all twenty encoded features after training-only imputation and standardization. We scale binary features too, which gives them comparable variance but can amplify rare indicators. The first two components retain only 20.95 percent of variance. Sixteen components are needed for ninety-five percent. PC1 combines higher salary and larger descriptions against Git. Actual record 90000480 is a low-salary Junior at negative PC1. Record 90000654 has a Lead title and high salary but a Junior heuristic label because of experience-rule priority. This explains some overlap. A fold-fitted PCA plus LR check reduces CV F1 to 0.7905; binary scaling changes too, so PCA alone cannot explain the whole difference.''',
'''We tested six MLP configurations covering one or two hidden layers, ReLU or tanh, regularization, learning rate and iteration limits. The selected network has thirty-two ReLU units with alpha 0.1 and learning rate 0.001. Scaling is learned within training folds. For learning progress, we separately divide only the training data into eight hundred fitting rows and two hundred validation rows before fitting preprocessing. The curve shows falling loss but a generalization gap: training F1 reaches 0.8996 while validation reaches 0.8253 at epoch two hundred. The final selected model hits its two-hundred-fifty iteration cap, so convergence is uncertain. Test F1 0.8093 is below Logistic Regression.''',
'''This chart compares all six models with the same macro F1 definition and saved test rows. Macro F1 gives each class equal weight, unlike accuracy which can hide a weak class. The dummy predicts only Middle and scores just 0.1822. Boosting is the CV-selected preferred model, and also has the highest test F1. Its weakest class is Senior: F1 0.8276 and recall 0.7792. Missing salary is an important error pattern: nineteen errors among sixty rows without salary, compared with nineteen among one hundred ninety with salary. Every actual error is saved with its synthetic ID and features, and all models have confusion matrices and classification reports. Titles appear in error tables for explanation only; they are never Scenario B inputs.''',
'''Our conclusion is that both ensembles improve on the corrected Midterm baseline within this synthetic experiment, while the MLP does not. PCA reveals template and compensation structure but loses too much information in two dimensions to prove seniority groups. For the Final we will use the descriptions already attached to these same synthetic IDs, mask seniority words and copied titles, and compare fold-fitted text and tabular models. Synthetic IDs must never be treated as real HeadHunter identities. Any later real-data study needs lawful sourcing and independently checked labels as a separate benchmark. OpenAI Codex implemented this continuation under Diyar’s direction; human review and the partner’s new contribution still require verification.''']
    evidence=[]
    for i,title in enumerate(titles):
        slide=prs.slides.add_slide(prs.slide_layouts[6]); slide.background.fill.solid(); slide.background.fill.fore_color.rgb=RGBColor(248,250,252)
        def text(x,y,w,h,txt,size=22,color=ink,bold=False):
            box=slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf=box.text_frame; tf.word_wrap=True
            for k,line in enumerate(txt.split('\n')):
                p=tf.paragraphs[0] if k==0 else tf.add_paragraph(); p.text=line; p.font.name='Aptos'; p.font.size=Pt(size); p.font.color.rgb=color; p.font.bold=bold; p.space_after=Pt(15)
            return box
        text(.65,.4,12,0.7,title,34,navy,True)
        text(.65,6.98,11,.3,'AITU · Machine Learning Algorithms · Synthetic snapshot / heuristic labels',11,accent)
        text(12.1,6.98,.6,.3,str(i+1),11,accent)
        if i==0:
            text(.8,1.65,11.8,1.4,copy[i][0],42,navy,True)
            text(.8,3.2,11.8,3,'\n'.join(copy[i][1:]),23)
        elif i in figures:
            text(.65,1.45,4.4,5.2,'\n'.join(copy[i]),21)
            slide.shapes.add_picture(str(ROOT/'results/figures'/figures[i]), Inches(5.25), Inches(1.55),width=Inches(7.5))
        else:
            text(.8,1.55,11.7,5.2,'\n'.join(copy[i]),26)
        speaker='Diyar' if i<4 else 'Galymzhan (proposed; verify)'
        notes=f'{speaker} | {seconds[i]} seconds\n{scripts[i]}\nEvidence: results/tables/model_comparison.csv, results/selection.json, CONCLUSIONS.md. AI assistance: OpenAI Codex.'
        slide.notes_slide.notes_text_frame.text=notes
        evidence.append({'slide':i+1,'title':title,'seconds':seconds[i],'speaker':speaker,'text':copy[i],'figure':figures.get(i),'evidence':['results/tables/model_comparison.csv','results/selection.json','CONCLUSIONS.md']})
    prs.save(ROOT/'presentation/Endterm_IT_Vacancy_Seniority.pptx')
    write('presentation/slide_evidence.json',json.dumps(evidence,indent=2))
    write('presentation/SCRIPT.md','# Seven-minute defense script\n\nTarget allocation: 420 seconds. Rehearse with a timer; actual delivery is unverified. Diyar: slides 1–4; Galymzhan: slides 5–8, proposed and subject to verification. Speaking assignment does not imply implementation authorship. Slow down at figures; use each slide’s full slot, including pointing and pauses.\n\n'+'\n\n'.join(f'## Slide {i+1}: {titles[i]} — {seconds[i]} seconds\nSpeaker: '+('Diyar' if i<4 else 'Galymzhan (proposed)')+'\n\n'+s for i,s in enumerate(scripts)))

if __name__=='__main__': build()

"""Build an editable, evidence-based deck and defense notes for the real benchmark."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from src.real_experiments import ROOT, OUT, LABELS, SOURCE, REV, NUM, CAT

NAVY='17324D';TEAL='008C95';GOLD='E9A23B';INK='25364A';GRAY='66788A';BG='F7FAFC'
def rgb(x):return RGBColor.from_string(x)
def read(name):return json.loads((OUT/name).read_text())
def table(df):return df.to_markdown(index=False) if False else '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in r)+' |' for r in df.itertuples(index=False,name=None)])

class Deck:
    def __init__(self):
        self.prs=Presentation();self.prs.slide_width=Inches(13.333);self.prs.slide_height=Inches(7.5);self.scripts=[];self.evidence=[]
    def text(self,s,x,y,w,h,txt,size=20,color=INK,bold=False):
        box=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame;tf.word_wrap=True
        for i,line in enumerate(txt.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color);p.space_after=Pt(10)
        return box
    def slide(self,title,script,evidence):
        s=self.prs.slides.add_slide(self.prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(BG)
        self.text(s,.6,.35,12.1,.85,title,32,NAVY,True)
        self.text(s,.65,7.05,11,.25,'Real Djinni job postings · Employer-grouped evaluation · AITU',10,GRAY)
        self.text(s,12.1,7.05,.6,.25,str(len(self.prs.slides)),10,GRAY)
        s.notes_slide.notes_text_frame.text=script+'\n\nEvidence: '+evidence+'\nSource: '+SOURCE+'\nPaper: https://aclanthology.org/2024.unlp-1.2/'
        self.scripts.append((title,script));self.evidence.append({'slide':len(self.prs.slides),'title':title,'evidence':evidence})
        return s
    def insight(self,s,txt):self.text(s,.7,6.25,11.9,.7,txt,20,TEAL,True)
    def chart(self,s,categories,series,kind=XL_CHART_TYPE.COLUMN_CLUSTERED,x=.8,y=1.5,w=11.7,h=4.5,maxval=None):
        d=CategoryChartData();d.categories=list(map(str,categories))
        for name,vals in series:d.add_series(name,[float(v) for v in vals])
        c=s.shapes.add_chart(kind,Inches(x),Inches(y),Inches(w),Inches(h),d).chart
        c.has_legend=len(series)>1
        if c.has_legend:c.legend.position=XL_LEGEND_POSITION.BOTTOM;c.legend.include_in_layout=False
        c.chart_style=10
        for axis in [c.category_axis,c.value_axis]:
            axis.tick_labels.font.size=Pt(14);axis.tick_labels.font.name='Aptos'
        c.value_axis.minimum_scale=0
        if maxval is not None:c.value_axis.maximum_scale=maxval
        for i,ser in enumerate(c.series):
            ser.format.fill.solid();ser.format.fill.fore_color.rgb=rgb([TEAL,NAVY,GOLD][i%3])
            ser.format.line.color.rgb=rgb([TEAL,NAVY,GOLD][i%3])
        return c
    def box(self,s,x,y,w,h,txt,color=NAVY):
        sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(color);sh.line.fill.background()
        tf=sh.text_frame;tf.word_wrap=True
        for i,line in enumerate(txt.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(20);p.font.color.rgb=rgb('FFFFFF')
        return sh
    def arrow(self,s,x1,y1,x2,y2):
        sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=rgb(TEAL);sh.line.width=Pt(3)
        self.text(s,(x1+x2)/2-.13,(y1+y2)/2-.21,.4,.4,'›',24,TEAL,True)
    def chain(self,s,labels,y=2.2):
        n=len(labels);w=(11.9-(n-1)*.4)/n
        for i,label in enumerate(labels):
            x=.7+i*(w+.4);self.box(s,x,y,w,1.5,label,TEAL if i==n-1 else NAVY)
            if i<n-1:self.arrow(s,x+w,y+.75,x+w+.4,y+.75)

def build():
    audit=read('data_audit.json');sel=read('selection.json');mlp=read('mlp.json');pca=read('pca.json');preferred=sel['preferred_model']
    comp=pd.read_csv(OUT/'tables/model_comparison.csv');df=pd.read_csv(ROOT/'data/real/benchmark.csv');curve=pd.read_csv(OUT/'tables/mlp_learning.csv');var=pd.read_csv(OUT/'tables/pca_variance.csv');proj=pd.read_csv(OUT/'tables/pca_projection.csv')
    report=read('tables/'+preferred.replace(' ','_')+'_report.json');best=comp[comp.model==preferred].iloc[0];lr=comp[comp.model=='Logistic Regression'].iloc[0]
    d=Deck()
    s=d.slide('IT Vacancy Experience Classification',f'Good afternoon. We revised our project after the feedback about synthetic data. This version uses actual job postings from the published Djinni Recruitment Dataset. The task is to predict an employer-declared experience band from indirect vacancy features. We compare six classifiers, select the model using employer-grouped training cross-validation, and report results on companies excluded from training. We call the classes experience bands rather than claiming independently verified Junior, Middle and Senior labels.','data/real/source_metadata.json; results/real/data_audit.json')
    d.text(s,.8,1.75,11.8,1.5,'Real job postings\nThree employer-declared experience bands',38,NAVY,True)
    d.text(s,.8,3.85,11.7,1.7,f'{audit["sample_rows"]:,} real records · {audit["train_rows"]:,} train / {audit["test_rows"]:,} test\nZholdubayev Diyar · Galymzhan Turemuratov',24)
    d.insight(s,'Updated after instructor feedback: synthetic results are historical only')
    s=d.slide('Dataset provenance',f'The source is the English job-description portion of the Djinni Recruitment Dataset, published by Drushchak and Romanyshyn at UNLP in 2024. The downloaded file contains {audit["raw_rows"]:,} rows. We pin the repository revision and record the file hash. After removing exact repeated descriptions, {audit["after_exact_dedup"]:,} records remain. We uniformly sample twelve thousand to bound training time. The vacancies are real; anonymized UUIDs identify corpus records and are not evidence of generated job content. No candidate CVs are used.','data/real/SOURCE_CARD.md; data/real/LICENSE; results/real/data_audit.json')
    d.chain(s,['Djinni platform\nReal postings','Published corpus\nUNLP 2024 / MIT',f'English file\n{audit["raw_rows"]:,} rows','Deduplicate\nSample 12,000'])
    d.text(s,.8,4.2,11.8,1.5,'Authors: Nazarii Drushchak and Mariana Romanyshyn\nPinned revision: '+REV[:12]+' · SHA-256 recorded\nSource paper: aclanthology.org/2024.unlp-1.2',21)
    s=d.slide('Target: declared experience requirements','The original target is the employer-provided Exp Years field. We map no experience and one year to the first band, two and three years to the second band, and five years to the third band. These are explicit operational categories, not a claim that five years always means Senior. Unlike the previous experiment, no title-based rule creates labels. We exclude the source target column, title and company identity from model inputs. The chart shows the class distribution in the actual sampled benchmark.','data/real/benchmark.csv; src/real_experiments.py:MAPPING')
    counts=df.target.value_counts();d.chart(s,LABELS,[('Real records',[counts.get(c,0) for c in LABELS])]);d.insight(s,'no_exp / 1y → 0–1 years     ·     2y / 3y → 2–3 years     ·     5y → 5+ years')
    s=d.slide('Vacancy categories', 'The corpus covers different technology and job categories. This chart shows the ten most frequent categories in our real sample. Category is an employer-provided input, not the experience label. We also use declared English level. We do not have reliable structured salary fields in this source, so we remove the salary pipeline rather than invent salaries or carry over synthetic values. These changes mean the new scores must be interpreted within this new benchmark and cannot be compared directly with old synthetic scores.','data/real/benchmark.csv')
    ct=df['Primary Keyword'].value_counts().head(10);d.chart(s,ct.index,[('Vacancies',ct.values)],XL_CHART_TYPE.BAR_CLUSTERED);d.insight(s,'Real category mix; no synthetic salaries or generated vacancy rows')
    s=d.slide('Declared English requirements','The English-level field is a real categorical requirement in this source. Missing values are explicitly represented as Unknown rather than guessed. The chart describes the sample, not a causal relationship between language and experience. One-hot encoding learns the available categories within each training fold and ignores unseen categories at validation time. This field may reflect regional and employer preferences, so its predictive usefulness should not be interpreted as a universal measure of professional seniority.','data/real/benchmark.csv; src/real_experiments.py:preprocessing')
    eng=df['English Level'].value_counts();d.chart(s,eng.index,[('Vacancies',eng.values)]);d.insight(s,'Missing requirements remain Unknown; categories are learned inside training folds')
    s=d.slide('Employer-grouped evaluation architecture',f'We first remove exact repeated descriptions. A fixed group split produces {audit["train_rows"]:,} training rows from {audit["train_companies"]:,} company groups and {audit["test_rows"]:,} test rows from {audit["test_companies"]:,} groups. No normalized company name appears on both sides. Five-fold cross-validation also keeps employer groups apart. The preferred model is fixed by mean CV macro F1 before test predictions. This is stricter than a random row split, but company-name normalization cannot perfectly resolve corporate aliases, and near-duplicate templates may remain.','data/real/train_ids.csv; data/real/test_ids.csv; results/real/verification.json')
    d.chain(s,['Deduplicated\nreal records','Employer split\nfixed seed 42','5-fold grouped CV\ntraining only','Freeze selection\nEvaluate test'])
    d.text(s,.9,4.25,11.6,1.5,f'Train: {audit["train_rows"]:,} rows / {audit["train_companies"]:,} employers\nTest: {audit["test_rows"]:,} rows / {audit["test_companies"]:,} employers\nEmployer overlap = 0 · Exact description overlap = 0',23)
    s=d.slide('Preprocessing and feature architecture','Our model input contains twenty-nine numeric features: description length, word count, technology count and twenty-six fixed technology indicators. We detect technology names with word boundaries, avoiding the old Go-inside-Django substring problem. Primary job category and English level are one-hot encoded. Imputation, scaling and category vocabularies are fitted inside each CV fold. Raw title, full text, company identity and the source experience target are excluded. Raw text still influences length and technology counts, so these features are indirect signals rather than proof of semantic understanding.','src/real_experiments.py:NUM,CAT,preprocessing')
    d.chain(s,['Description\nCounts + 26 flags','Numeric pipeline\nImpute + scale','Combine with\ncategory + English','Classifier\n3 experience bands'])
    d.text(s,.8,4.25,11.7,1.4,'Fold training rows: fit preprocessing and classifier\nFold validation / final test: transform and predict only\nExcluded: Exp Years, title, employer, ID, full description',22)
    s=d.slide('Six-model comparison',f'The chart compares mean employer-grouped training CV macro F1 with test macro F1. Macro F1 gives equal weight to each experience band. The preferred model is {preferred}, with mean CV F1 {best.cv_mean:.4f} and fold standard deviation {best.cv_std:.4f}. Its held-out F1 is {best.test_macro_f1:.4f}. The standard deviation describes variation across folds and is not a confidence interval. We do not claim statistical significance or a globally optimal parameter configuration. These real-data results replace the synthetic numbers for the current defense.','results/real/tables/model_comparison.csv; results/real/selection.json')
    d.chart(s,comp.model,[('Grouped CV',comp.cv_mean),('Held-out employers',comp.test_macro_f1)],maxval=1);d.insight(s,f'Preferred by training CV: {preferred} · Test macro F1 {best.test_macro_f1:.3f}')
    s=d.slide('Ensemble architectures','Random Forest trains randomized trees on bootstrap samples and combines their votes. Our grid varies maximum depth and minimum leaf size, with one hundred and fifty trees fixed. Histogram Gradient Boosting instead adds trees sequentially to reduce the current loss. Its grid varies leaf complexity and L2 regularization, with one hundred and fifty iterations fixed. Each model has four configurations evaluated on the same five grouped folds. The smaller grids bound computation; selection still uses training data only.','results/real/selection.json; src/real_experiments.py:specs')
    d.text(s,.8,1.45,11.7,.5,'Random Forest: independent randomized trees → combined votes',23,NAVY,True)
    d.chain(s,['Bootstrap trees\n150 estimators','Parallel tree\npredictions','Aggregate votes\nClass decision'],y=2.15)
    d.text(s,.8,4.05,11.7,.5,'Histogram Boosting: sequential corrections → additive model',23,NAVY,True)
    d.chain(s,['Current model\nCompute loss','New tree\nGradient correction','Updated ensemble\nRepeat 150 rounds'],y=4.7)
    s=d.slide(f'{preferred}: confusion matrix',f'This matrix uses only held-out employers. Rows are actual experience bands and columns are predictions. Diagonal entries are correct predictions. The preferred model makes {int(best.errors)} errors among {audit["test_rows"]:,} records. Off-diagonal counts show which requirements are confused. These are mistakes against employer-declared categories, not against an independently audited measure of professional competence. Every prediction is saved with the original corpus ID for inspection.','results/real/tables/'+preferred.replace(' ','_')+'_confusion.csv; results/real/tables/test_predictions.csv')
    cm=pd.read_csv(OUT/'tables'/f'{preferred.replace(" ","_")}_confusion.csv',index_col=0)
    t=s.shapes.add_table(4,4,Inches(1.4),Inches(1.65),Inches(10.5),Inches(4.2)).table
    for r in range(4):
        for c in range(4):
            cell=t.cell(r,c);cell.text='True / predicted' if r==c==0 else (LABELS[c-1] if r==0 else LABELS[r-1] if c==0 else str(cm.iloc[r-1,c-1]))
            cell.fill.solid();cell.fill.fore_color.rgb=rgb(NAVY if r==0 or c==0 else 'C4E9E9' if r==c else 'E8EFF5')
            for p in cell.text_frame.paragraphs:p.font.size=Pt(24);p.font.color.rgb=rgb('FFFFFF' if r==0 or c==0 else INK)
    d.insight(s,f'{int(best.errors):,} errors / {audit["test_rows"]:,} real test vacancies · All prediction rows saved')
    s=d.slide('Performance by experience band','This chart separates precision, recall and F1 for each band. Precision asks how often a prediction of the band is correct. Recall asks how many actual members of that band are found. A strong overall accuracy can hide a weak class, which is why macro F1 is primary. Class difficulty must be read from the new real-data report; we cannot reuse the old statement about the hardest synthetic class.','results/real/tables/'+preferred.replace(' ','_')+'_report.json')
    d.chart(s,LABELS,[(name,[report[c][key] for c in LABELS]) for name,key in [('Precision','precision'),('Recall','recall'),('F1','f1-score')]],maxval=1);d.insight(s,'Class reports expose errors hidden by overall accuracy')
    s=d.slide('MLP architecture',f'The neural network uses {mlp["input_features"]} encoded inputs, one hidden layer with thirty-two ReLU units and a three-class softmax output. Adam optimizes the weights. Two L2 regularization values are compared using grouped CV. The final fit performs {mlp["iterations"]} iterations out of a maximum of {mlp["max_iter"]}. Reaching the cap means convergence is not established. A separate grouped training-only diagnostic tracks learning without looking at the final test set.','results/real/mlp.json; results/real/selection.json')
    d.chain(s,[f'{mlp["input_features"]} inputs\nScaled + one-hot','32 hidden units\nReLU','3 output classes\nSoftmax'])
    d.text(s,.85,4.3,11.7,1.4,f'Optimizer: Adam · Learning rate: 0.001 · Selected alpha: {mlp["alpha"]}\nFinal fit: {mlp["iterations"]}/{mlp["max_iter"]} iterations · Hit cap: {mlp["hit_cap"]}\nTitle and experience target are excluded from inputs',23)
    s=d.slide('MLP learning progress',f'The diagnostic holds out employer groups from the training set: {mlp["diagnostic_fit_rows"]} fitting rows and {mlp["diagnostic_validation_rows"]} validation rows. We fit preprocessing only on the fitting partition and record F1 over one hundred epochs. Final diagnostic training F1 is {curve.iloc[-1].train_f1:.3f} and validation F1 is {curve.iloc[-1].validation_f1:.3f}. The curves help assess generalization and optimization, but this diagnostic network is separate from the final full-training fit. No test-guided stopping is used.','results/real/tables/mlp_learning.csv; results/real/mlp.json')
    d.chart(s,curve.epoch,[('Training F1',curve.train_f1),('Grouped validation F1',curve.validation_f1)],XL_CHART_TYPE.LINE,maxval=1);d.insight(s,'Independent diagnostic inside training data · Final test excluded')
    s=d.slide('PCA projection of training vacancies',f'PCA is fitted only on encoded training features and does not use experience labels. The first two components retain {pca["first_two"]:.1%} of variance. The scatter colors are added afterward for interpretation. Overlap does not imply that the task is impossible: the projection discards information. Apparent separation also does not prove natural seniority groups. PCA is an explanatory diagnostic and is not part of the six-model classification pipelines.','results/real/tables/pca_projection.csv; results/real/pca.json')
    xy=XyChartData()
    sample=proj.sample(n=min(1200,len(proj)),random_state=42)
    for c in LABELS:
        ser=xy.add_series(c)
        for r in sample[sample.target==c].itertuples():ser.add_data_point(float(r.pc1),float(r.pc2))
    chart=s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER,Inches(.8),Inches(1.5),Inches(11.7),Inches(4.5),xy).chart;chart.has_legend=True;chart.legend.position=XL_LEGEND_POSITION.BOTTOM
    d.insight(s,f'Two components retain {pca["first_two"]:.1%} variance · Display: up to 1,200 training points')
    s=d.slide('PCA explained variance',f'The cumulative curve quantifies information retained by the projection. In this benchmark, {pca["components95"]} components are needed to reach ninety-five percent of variance out of {pca["features"]} encoded features. The numerical feature block is standardized within training; categorical indicators are one-hot encoded. This is a different representation from the old synthetic experiment. Explained variance is not classification accuracy, so we do not interpret ninety-five percent retained variance as ninety-five percent correct predictions.','results/real/tables/pca_variance.csv; results/real/pca.json')
    d.chart(s,var.component,[('Cumulative variance',var.cumulative)],XL_CHART_TYPE.LINE,maxval=1);d.insight(s,f'{pca["components95"]} components needed for 95% of encoded training variance')
    s=d.slide('Conclusions and remaining limitations',f'The current result is a real-data benchmark with {preferred} preferred by grouped training CV and test macro F1 {best.test_macro_f1:.4f}. We keep the synthetic experiment as historical evidence but do not compare its scores directly with the new task. The labels describe employer requirements rather than verified ability, the sample is limited to English Djinni postings, and near-duplicate text or employer aliases may remain. The next steps are a larger sample, stronger duplicate grouping and a chronological external evaluation. We will report improvements or failures using the same fixed evaluation protocol. Thank you.','results/real/selection.json; results/real/data_audit.json; results/real/verification.json')
    d.text(s,.85,1.65,11.8,1.1,f'Preferred model: {preferred}\nTest macro F1: {best.test_macro_f1:.4f} · Accuracy: {best.test_accuracy:.4f}',28,NAVY,True)
    d.text(s,.85,3.25,11.8,2.6,'Employer-declared experience ≠ independently verified skill level\nEnglish Djinni sample; limited geographic and historical coverage\nExact duplicates removed; near-duplicates and employer aliases may remain\nNext: larger sample, template grouping, chronological evaluation',22)
    d.insight(s,'New dataset, new split, new task definition: old synthetic scores do not transfer')
    # Retain the upstream license discrepancy instead of misnaming the downloaded notice.
    d.prs.slides[1].notes_slide.notes_text_frame.text += '\nLicense clarification: the dataset card/paper declares MIT, while the upstream repository LICENSE downloaded at preparation time is Apache-2.0. Both notices are retained; no license reconciliation is claimed.'
    dest=ROOT/'presentation/IT_Vacancies.pptx';d.prs.save(dest)
    (ROOT/'presentation/SCRIPT.md').write_text('# Real-data defense script\n\nApproximate target: 9–11 minutes; rehearse with the actual slides. Speaking roles should reflect actual contributions.\n\n'+'\n\n'.join(f'## Slide {i+1}: {title}\n\n{script}' for i,(title,script) in enumerate(d.scripts)))
    (ROOT/'presentation/real_slide_evidence.json').write_text(json.dumps(d.evidence,indent=2))
    summary=table(comp)
    (ROOT/'README.md').write_text(f'''# Real IT Vacancy Experience Classification

The benchmark uses **real Djinni job postings** from the published recruitment corpus.

Source: [Djinni English job descriptions]({SOURCE}), revision `{REV}`. Authors: Nazarii Drushchak and Mariana Romanyshyn; [UNLP 2024 paper](https://aclanthology.org/2024.unlp-1.2/). MIT license and original source card are saved in `data/real/`. UUIDs are anonymized identifiers; vacancy texts are real records from the published corpus. No candidate CVs are used.

## Current experiment

- Downloaded source: {audit['raw_rows']:,} English records; {audit['after_exact_dedup']:,} after filtering and exact description deduplication.
- Uniform seed-42 sample: {audit['sample_rows']:,} records. Saved employer-grouped split: {audit['train_rows']:,} train / {audit['test_rows']:,} test.
- Targets: employer `Exp Years`: no_exp/1y → 0–1 years; 2y/3y → 2–3 years; 5y → 5+ years. These are experience bands, not verified Junior/Middle/Senior labels.
- Inputs: 29 numeric aggregate/technology features plus category and English level. No title, employer, ID or experience target enters the model. No structured salary is available.
- Five employer-grouped training folds; preprocessing fitted inside each fold. Preferred model chosen before held-out predictions. No exact description or normalized employer overlap in train/test.

{summary}

## Run

```sh
python -m pip install -r requirements.txt
python -m src.real_experiments
python -m src.build_real_deliverables
```

Python 3.15 in the project's configured environment was used. Source Parquet must be at `data/real/djinni_jobs.parquet`; pinned download URL is in `data/real/README.md`. Results and fitted pipelines: `results/real/`. Benchmark rows and split IDs: `data/real/`. Runtime is several minutes; a uniform 12,000-row sample bounds compute. This is not a full-corpus estimate.

## Defense artifacts

- [New editable presentation](presentation/IT_Vacancies.pptx): 16 slides, 8 native charts, editable architecture diagrams and confusion matrix.
- [English oral script](presentation/SCRIPT.md).
- [Russian explanation and English Q&A](DEFENSE_GUIDE.md).
- [New conclusions](CONCLUSIONS.md), [migration changelog](CHANGELOG.md).

Employer labels can be noisy; near-duplicates and employer aliases may survive. CV spread is not a confidence interval; selected CV scores are optimistic, and no significance claim is made.
''')
    (ROOT/'CONCLUSIONS.md').write_text(f'''# Findings on real Djinni vacancies

{summary}

Preferred by employer-grouped training CV: **{preferred}**. Its test macro F1 is {best.test_macro_f1:.4f}, accuracy {best.test_accuracy:.4f}, errors {int(best.errors)}/{audit['test_rows']}. Corrected LR test F1 is {lr.test_macro_f1:.4f}; differences on this split are descriptive, not significance claims.

PCA: first two components retain {pca['first_two']:.2%}; {pca['components95']} components reach 95% out of {pca['features']}. PCA is exploratory, not part of the classifier ranking. Numeric features are standardized; categorical inputs are one-hot encoded.

MLP: {mlp['input_features']} inputs → 32 ReLU → 3 softmax outputs. Selected alpha {mlp['alpha']}; final iterations {mlp['iterations']}/{mlp['max_iter']}; hit cap: {mlp['hit_cap']}. The grouped training-only diagnostic ends with train F1 {curve.iloc[-1].train_f1:.4f}, validation F1 {curve.iloc[-1].validation_f1:.4f}. This is a distinct fit from the final model.

Interpretation is restricted to the uniformly sampled English-language Djinni corpus and employer-declared experience bands. Exact descriptions are deduplicated before splitting; normalized employer groups do not overlap. Alias resolution and near-duplicate/template detection remain imperfect. The sample is not geographically representative, and random employer holdout does not measure future-time performance. No external validity or causal claim is made.

Next: larger sample, near-duplicate connected-component grouping, chronological holdout and independent label audit. Old synthetic metrics are historical and not directly comparable with this new task.
''')
    (ROOT/'CHANGELOG.md').write_text('''# Migration after instructor feedback

The instructor rejected generated vacancy records. The current revision replaces them with published real Djinni job postings. This explicitly supersedes the old same-dataset/same-split continuity assumption for the new benchmark.

| Previous version | Current version | Reason / evidence |
| --- | --- | --- |
| 1,250 generated vacancies | Published real English Djinni corpus, deterministic 12,000-row sample | data/real/source_metadata.json, SOURCE_CARD.md, LICENSE |
| Title/experience heuristic Junior/Middle/Senior | Employer-declared experience bands: 0–1, 2–3, 5+ years | src/real_experiments.py:MAPPING |
| Saved synthetic row split | New fixed employer-grouped split with zero normalized employer overlap | data/real/train_ids.csv, test_ids.csv |
| Salary and synthetic skill fields | Description aggregates, 26 bounded technology flags, category and English level | Real source has no structured salary field |
| Substring skill extraction | Word-boundary matching | Go no longer matches Django |
| Stratified row CV | Five stratified employer-grouped training folds | results/real/verification.json |
| Synthetic results and eight-slide deck | Recomputed real-data results and 16-slide visual deck | results/real/; presentation/IT_Vacancies.pptx |
| Old defense narrative | Updated explanation and English oral script | DEFENSE_GUIDE.md; presentation/SCRIPT.md |

Preprocessing remains inside model pipelines. Source file revision and SHA-256, split IDs, all test predictions, per-model reports, CV candidates and fitted pipelines are saved. Replacing data changes the task: scores across the two versions cannot establish model improvement.

The new requirement comes from the user's report of instructor feedback; acceptance of the revised task by the instructor has not been independently confirmed.
''')
    (ROOT/'RUBRIC_CHECKLIST.md').write_text('''# Current real-data evidence

| Area | Evidence | Limit |
| --- | --- | --- |
| Real provenance | data/real/source_metadata.json, SOURCE_CARD.md, LICENSE | Published secondary source; not a new collection |
| Evaluation | Saved employer split; 5 grouped CV folds | Not temporal or external evaluation |
| Ensembles | RF and histogram boosting, 4 candidates each | Bounded grids, no nested CV |
| Neural network | 2 regularization settings, learning curve, convergence record | Check iteration cap |
| Unsupervised analysis | Train-only PCA, loadings, projection, variance | Not clustering evidence |
| Error analysis | All held-out predictions, reports and confusion matrices | Employer labels are not independent truth |
| Visual presentation | 16 slides; editable native charts, architectures and matrix | Speaking duration must be rehearsed |
| Dataset revision | Published Djinni corpus and employer-grouped split | Scores from different datasets are not directly comparable |

This is evidence mapping, not a guaranteed grade. Actual individual contributions must be stated accurately.
''')
    scripts=(ROOT/'presentation/SCRIPT.md').read_text()
    guide=f'''# Real Djinni data — объяснение и защита

**Эта версия заменяет объяснение синтетического эксперимента.** Основной результат теперь рассчитан на реальных вакансиях. Старые 0.8478 macro F1 и 84.8% accuracy нельзя использовать для новой версии.

## 1. Что поменяли после замечания преподавателя

Вместо генератора скачан опубликованный авторами Djinni Recruitment Dataset. Используется только англоязычная часть вакансий; CV кандидатов не загружаются. Источник: [карточка]({SOURCE}), [научная статья UNLP 2024](https://aclanthology.org/2024.unlp-1.2/). Авторы — Nazarii Drushchak и Mariana Romanyshyn. Лицензия MIT сохранена рядом с исходным файлом.

Исходный Parquet содержит **{audit['raw_rows']:,}** записей. После фильтрации и удаления одинаковых нормализованных описаний осталось **{audit['after_exact_dedup']:,}**. Для ограниченного времени обучения взята равномерная случайная выборка **12 000** реальных записей с seed 42. Это подвыборка, а не результат на полном корпусе.

UUID у записей анонимизированы авторами корпуса. Анонимизация ключа не означает, что текст вакансии сгенерирован. Мы не используем UUID как признак и не пытаемся связать его с живой вакансией.

## 2. Что предсказываем

Источник содержит `Exp Years` — требование работодателя к опыту. Отображение классов фиксировано до обучения:

| Значение источника | Класс модели |
| --- | --- |
| no_exp или 1y | 0–1 years |
| 2y или 3y | 2–3 years |
| 5y | 5+ years |

Эти диапазоны нельзя без оговорки называть проверенными Junior/Middle/Senior. Seniority зависит не только от стажа. Новая метка не вычисляется из названия: это укрупнение исходного требования работодателя. `Exp Years` используется только как источник y и исключён из X.

## 3. Какие признаки используются

29 числовых столбцов: число символов, число слов, число найденных технологий и 26 индикаторов технологий. Две категории: `Primary Keyword` — специализация вакансии, `English Level` — требуемый английский. Число входов после one-hot на полном train — **{mlp['input_features']}**.

Технологии извлекаются фиксированными регулярными выражениями с границами слова. В отличие от старой версии, Go не находится внутри Django. Такой словарь всё ещё неполон: синонимы и альтернативные написания могут быть пропущены.

Название, работодатель, ID, исходный столбец опыта и полный текст не подаются классификатору. Из текста используются только счётчики и флаги технологий. Текстовые требования могут косвенно влиять на эти агрегаты, поэтому полное отсутствие любых косвенных подсказок не заявляется.

В этом источнике нет структурированной зарплаты. Мы убрали зарплатные признаки и старый анализ missing salary, вместо того чтобы придумывать значения. Поэтому постановка и состав признаков отличаются от Midterm.

## 4. Разделение и защита от утечки

Train: **{audit['train_rows']:,}** строк, **{audit['train_companies']:,}** нормализованных работодателей. Test: **{audit['test_rows']:,}** строк, **{audit['test_companies']:,}** работодателей. Размеры могут отличаться от ровно 80/20 по строкам: GroupShuffleSplit выделяет долю групп компаний.

Работодатель нормализуется приведением к нижнему регистру и удалением неалфавитно-цифровых разделителей. Это помогает объединить варианты написания, но не разрешает все юридические алиасы. Один нормализованный работодатель не попадает в обе части. Точные одинаковые описания удалены глобально до sampling и split; близкие шаблоны ещё могут остаться.

На train применяется StratifiedGroupKFold: пять fold, классы по возможности сбалансированы, компания не пересекает стороны fold. Pipeline изучает медианы, масштабы и one-hot словари только на обучающей стороне. Test используется после выбора модели по CV. Нет проверки на новых датах и нет внешнего независимого корпуса.

## 5. Модели и подбор параметров

- Dummy: всегда самый частый training-класс; минимальный ориентир.
- Logistic Regression: линейная многоклассовая модель, max_iter 1500, C=1.
- KNN: 15 соседей, голоса взвешены по расстоянию; чувствителен к масштабам и one-hot расстояниям.
- Random Forest: 150 деревьев; 4 комбинации глубины 12/без ограничения и минимального листа 2/8.
- HistGradientBoosting: 150 последовательных итераций; 4 комбинации числа листьев 7/15 и L2 1/5.
- MLP: один скрытый слой 32 ReLU; Adam; max_iter 180; alpha 0.1/1.0 — две CV-конфигурации.

Выбранные параметры находятся в `results/real/selection.json`. Ограниченная сетка не доказывает глобальный оптимум. Выбор по одному и тому же CV даёт оптимизм; nested CV не проводился.

## 6. Метрики и новые результаты

Precision — доля правильных ответов среди предсказаний класса. Recall — доля найденных объектов этого класса. F1 — гармоническое среднее precision и recall. Macro F1 — обычное среднее F1 трёх классов; оно не даёт частому классу скрыть слабые результаты остальных. Accuracy — доля всех верных ответов.

{summary}

Выбрана **{preferred}** по среднему training CV macro F1 **{best.cv_mean:.4f}**. CV std **{best.cv_std:.4f}** отражает разброс пяти fold, не доверительный интервал. Test macro F1 **{best.test_macro_f1:.4f}**, accuracy **{best.test_accuracy:.4f}**, ошибок **{int(best.errors)}/{audit['test_rows']}**. Это новые расчёты; старые synthetic-числа не перенесены.

Не утверждаем статистическую значимость отличий между моделями. Разницу между прежним и новым датасетом нельзя называть улучшением или ухудшением алгоритма: изменились данные, метки, признаки и разбиение.

## 7. MLP: архитектура и обучение

**{mlp['input_features']} входов → 32 ReLU → 3 softmax выхода**. Финальная модель выполнила **{mlp['iterations']}/{mlp['max_iter']}** итераций, достигла лимита: **{mlp['hit_cap']}**. При достижении лимита сходимость не подтверждена.

Learning curve строится отдельной моделью внутри train: {mlp['diagnostic_fit_rows']} строк для fit и {mlp['diagnostic_validation_rows']} для validation, компании разделены. Preprocessing fit только на fit-части. На 100-й эпохе train F1 **{curve.iloc[-1].train_f1:.4f}**, validation F1 **{curve.iloc[-1].validation_f1:.4f}**. Разрыв оценивается по кривым; диагностическая модель не равна финальной модели. Test не используется для early stopping.

## 8. PCA

PCA обучается без y на закодированном train. Числовые признаки стандартизированы, категории представлены one-hot. Первые две компоненты сохраняют **{pca['first_two']:.2%}** дисперсии; для 95% нужны **{pca['components95']}** из **{pca['features']}** компонент. Цвета классов добавлены после fit.

PCA ищет дисперсию, а не классы. Проекция не доказывает существование трёх естественных групп. PCA не входит в основной pipeline классификации. На слайде показано до 1200 training-точек для читаемости; компоненты обучены на всём train.

## 9. Какие визуальные материалы добавлены

В [новой презентации](presentation/IT_Vacancies.pptx) **16 слайдов**. Распределение классов и категорий, сравнение моделей, метрики классов, learning curve и PCA — редактируемые диаграммы. Confusion matrix — редактируемая таблица. Схемы происхождения данных, grouped evaluation, preprocessing, ансамблей и MLP состоят из редактируемых объектов PowerPoint. Источники и объяснения сохранены в notes каждого слайда.

Каждый график отвечает на вопрос: что за данные, как оцениваем, какая модель лучше по выбранному критерию, где ошибки, как обучается сеть, сколько информации сохраняет PCA.

## 10. Ограничения, которые стоит сказать вслух

Это реальный опубликованный корпус, но не репрезентативный снимок всего рынка. Подвыборка англоязычная, историческая и ограничена одной платформой. Требование работодателя не является независимой экспертизой уровня. Близкие тексты и алиасы компаний могут пересекаться. Результат на новых компаниях не равен прогнозу на будущие годы. Авторство работы нужно описывать по фактическому участию; роли на защите не доказывают личный вклад.

## 11. English oral defense

Полный текст слайд за слайдом также находится в [SCRIPT.md](presentation/SCRIPT.md). Ориентир 9–11 минут; темп необходимо проверить репетицией.

'''+scripts.replace('# Real-data defense script','### Full script',1)+f'''

## 12. Questions and answers in English

**Are these vacancies generated?** No. We use actual postings from the published Djinni corpus. We generate neither vacancy texts nor additional training rows. The corpus uses anonymized UUIDs for record identity.

**Where is the evidence?** The dataset has a UNLP 2024 publication, a public repository, a source card and an MIT license. We saved the repository revision and SHA-256 of the downloaded file.

**Are the labels Junior, Middle and Senior?** We now predict declared experience bands: zero to one, two to three, and five or more years. These are based on the employer's structured requirement. They are not independently verified seniority labels.

**Why did you change the task name?** The real source provides required experience rather than expert seniority annotations. Naming the target accurately is more defensible than forcing unsupported Junior/Middle/Senior labels.

**Did you keep the old split?** No. Its IDs belong to synthetic vacancies. The instructor's request requires a new dataset and split. We save the real-data split explicitly.

**How do you prevent company memorization?** Normalized employer names are disjoint across train and test, and across the training sides of CV folds. Employer identity is not a feature. Near-duplicate templates and unresolved company aliases remain limitations.

**Why only twelve thousand records?** We use a uniform reproducible sample to bound grouped grid-search computation. All sampled records are real. We disclose the sample size and do not claim a full-corpus result.

**Why no salary features?** This source has no structured salary field. We remove those features rather than generate replacement values.

**Which model wins?** {preferred} is preferred by mean grouped training CV macro F1. Its test macro F1 is {best.test_macro_f1:.4f}. The test is used for reporting after selection.

**Why might the result be lower than before?** The real-data task has different labels, features, company grouping and distributions. The scores are not directly comparable. Real-data credibility does not require retaining the old synthetic score.

**Is PCA a classifier?** No. It is an unsupervised projection used here to inspect variance. It does not optimize label separation and is not part of the primary classifier ranking.

**What remains uncertain?** Independent label validity, representativeness, near-duplicate leakage, employer alias resolution and performance on later dates or another platform remain unverified.

## 13. Files to show the instructor

- Provenance: [source card](data/real/SOURCE_CARD.md), [MIT license](data/real/LICENSE), [pinned metadata](data/real/source_metadata.json).
- Data and split audit: [data_audit.json](results/real/data_audit.json).
- Selection criterion and parameters: [selection.json](results/real/selection.json).
- All results: [model_comparison.csv](results/real/tables/model_comparison.csv).
- All held-out predictions: [test_predictions.csv](results/real/tables/test_predictions.csv).
- Errors: [preferred_errors.csv](results/real/tables/preferred_errors.csv).
- Architecture and training code: [real_experiments.py](src/real_experiments.py).
- Runtime integrity checks: [verification.json](results/real/verification.json).
'''
    guide=guide.replace('Лицензия MIT сохранена рядом с исходным файлом.','Карточка датасета и статья заявляют MIT. В LICENSE исходного репозитория при скачивании указан Apache-2.0; это расхождение сохранено явно, а не объявлено устранённым.')
    guide=guide.replace('[MIT license](data/real/LICENSE)','[upstream repository license](data/real/LICENSE)')
    guide=guide.replace('a source card and an MIT license.','a source card declaring MIT. The upstream repository LICENSE is Apache-2.0, and we disclose this discrepancy.')
    (ROOT/'DEFENSE_GUIDE.md').write_text(guide)
    for filename in ['README.md','CHANGELOG.md','RUBRIC_CHECKLIST.md']:
        path=ROOT/filename
        path.write_text(path.read_text().replace('MIT license and original source card are saved','The source card declares MIT; the upstream repository LICENSE is Apache-2.0. Both source notices are saved').replace('MIT notice is retained','Upstream license notice is retained').replace('MIT license |','source license notices |'))
    (ROOT/'DEFENSE_QA.md').write_text('# Real-data defense questions\n\nSee the current questions and answers in [DEFENSE_GUIDE.md](DEFENSE_GUIDE.md#12-questions-and-answers-in-english).\n')
    print('Built',len(d.prs.slides),'slides:',dest,flush=True)

if __name__=='__main__':build()

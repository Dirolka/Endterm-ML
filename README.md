# Real IT Vacancy Experience Classification

The benchmark uses **real Djinni job postings** from the published recruitment corpus.

Source: [Djinni English job descriptions](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english), revision `b56a6054c10f1266a141e37c8df66c79ff2863af`. Authors: Nazarii Drushchak and Mariana Romanyshyn; [UNLP 2024 paper](https://aclanthology.org/2024.unlp-1.2/). The source card declares MIT; the upstream repository LICENSE is Apache-2.0. Both source notices are saved in `data/real/`. UUIDs are anonymized identifiers; vacancy texts are real records from the published corpus. No candidate CVs are used.

## Current experiment

- Downloaded source: 141,897 English records; 141,897 after filtering and exact description deduplication.
- Uniform seed-42 sample: 12,000 records. Saved employer-grouped split: 9,906 train / 2,094 test.
- Targets: employer `Exp Years`: no_exp/1y → 0–1 years; 2y/3y → 2–3 years; 5y → 5+ years. These are experience bands, not verified Junior/Middle/Senior labels.
- Inputs: 29 numeric aggregate/technology features plus category and English level. No title, employer, ID or experience target enters the model. No structured salary is available.
- Five employer-grouped training folds; preprocessing fitted inside each fold. Preferred model chosen before held-out predictions. No exact description or normalized employer overlap in train/test.

| model | cv_mean | cv_std | test_macro_f1 | test_accuracy | errors |
| --- | --- | --- | --- | --- | --- |
| MLP | 0.4632 | 0.0152 | 0.4776 | 0.6208 | 794 |
| Boosting | 0.4594 | 0.0058 | 0.4573 | 0.6294 | 776 |
| Logistic Regression | 0.4418 | 0.0106 | 0.4578 | 0.6433 | 747 |
| Random Forest | 0.4113 | 0.0167 | 0.4373 | 0.6318 | 771 |
| KNN | 0.4018 | 0.0098 | 0.4278 | 0.6022 | 833 |
| Dummy | 0.2527 | 0.0001 | 0.2506 | 0.6022 | 833 |

## Run

```sh
python -m pip install -r requirements.txt
python -m src.real_experiments
python -m src.build_real_deliverables
```

Python 3.15 in the project's configured environment was used. Source Parquet must be at `data/real/djinni_jobs.parquet`; pinned download URL is in `data/real/README.md`. Results and fitted pipelines: `results/real/`. Benchmark rows and split IDs: `data/real/`. Runtime is several minutes; a uniform 12,000-row sample bounds compute. This is not a full-corpus estimate.

The executed real-data notebook is [Real_Djinni_Experience.ipynb](notebooks/Real_Djinni_Experience.ipynb). By default it inspects saved results and verifies all model predictions; set `REBUILD=True` to repeat training. A PDF copy is in `presentation/IT_Vacancies.pdf`.

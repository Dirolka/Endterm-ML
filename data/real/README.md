# Real Djinni job postings

Source: https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english

Pinned revision: b56a6054c10f1266a141e37c8df66c79ff2863af

Download (145.9 MB):

```sh
curl -fL 'https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english/resolve/b56a6054c10f1266a141e37c8df66c79ff2863af/data/train-00000-of-00001.parquet' -o data/real/djinni_jobs.parquet
```

Authors: Nazarii Drushchak and Mariana Romanyshyn. 2024. Introducing the Djinni Recruitment Dataset: A Corpus of Anonymized CVs and Job Postings. UNLP @ LREC-COLING, pages 8–13. https://aclanthology.org/2024.unlp-1.2/

Original repository: https://github.com/Stereotypes-in-LLMs/recruitment-dataset . The card declares MIT, while the upstream repository LICENSE is Apache-2.0; that original notice is retained in LICENSE. SOURCE_CARD.md is an unmodified downloaded source card. source_metadata.json records the pinned Hugging Face commit.

Use only job records; candidate CVs are neither downloaded nor included. UUIDs are anonymized corpus identifiers, not generated vacancy content or live platform IDs. No title-based labels or invented salary values are added.

Run `python -m src.real_experiments` to save a deterministic 12,000-row uniform sample after exact-description deduplication, new employer-grouped split IDs and the audit. The raw HF split named `train` is a storage partition, not our evaluation split. Our own employer split and grouped CV are constructed within it.

Targets explicitly aggregate employer Exp Years: no_exp/1y → 0–1 years, 2y/3y → 2–3 years, 5y → 5+ years. This is not expert annotation of Junior/Middle/Senior. Salary is absent. Do not join synthetic IDs from data/raw to this corpus.

Raw Parquet and benchmark CSV are local download artifacts excluded from Git by size. Reproduce them using the pinned URL and experiment command; split CSVs and audit remain small and reviewable. Do not publish additional mirrors of the corpus without preserving its original attribution and source license notices.

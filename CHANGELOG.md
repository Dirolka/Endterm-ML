# Migration after instructor feedback

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

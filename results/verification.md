# Execution and submission verification

- Read Endterm Rubric.pdf in full (all eight criteria). Available reference deck is IT Vacancy Seniority.pdf; the requested (1) variant is absent.
- Inspected original generator, collector, baseline, notebook builder, notebook, README, metric CSV, raw CSV/SQLite and split IDs. Raw data now live in data/raw/; source backups are in the ignored archive/midterm_originals.zip.
- Dataset: 1250 unique synthetic IDs; 1000 train and 250 test; no overlap, unknown or missing rows. CSV and SQLite match on all original raw fields. Class counts: Junior 396, Middle 470, Senior 384; 344 missing salaries; RUR throughout.
- Before cleanup, the initial SHA-256 inventory verified all eleven named original source/data files. That inventory is retained in the backup. The current data/original_sha256.txt and results/original_manifest.json cover the relocated raw data and historical baseline reference.
- Full command-line experiment completed; original metrics reproduced. results/execution.log retains actual output and convergence warnings.
- Endterm notebook executed through PyCharm's kernel, including the full training pipeline and visual outputs. All three code cells have execution counts and no error outputs. Original Midterm notebook is intentionally preserved, not rerun: its collection/split-writing behavior conflicts with immutable-data requirements.
- python -m pytest -q: 11 passed. results/test_results.txt.
- Fresh isolated environment in build/clean_room/.venv: requirements.txt installed successfully; full pipeline and tests passed; consolidated metrics match within 1e-10. results/clean_environment.log and clean_environment_verified.json. The clean-room experiment predates only the explicit default plot-theme fix, which changes rendering but not model calculations.
- Eight-slide PPTX generated using python-pptx as explicitly requested; speaker notes and a 967-word script with 420-second proposed allocation are included. No claim of actual timed rehearsal.
- Opened the generated PPTX in installed Microsoft PowerPoint. Exported locally using Optimize for printing to PDF; refreshed after the final figures/text. PDF has eight pages and renders with macOS PDFKit. Preview PNGs under presentation/previews/ were visually inspected; all slides have readable layout and no clipping. pypdf reports nonfatal dangling object-pointer warnings in the PowerPoint export; all pages and expected text still parse and PDFKit renders them.
- Final artifact audit recomputes every reported test metric from saved predictions, checks actual error IDs, preferred CV choice, slide metrics, notes, shape bounds, PDF titles/pages, notebook outputs and original integrity. results/artifact_audit.json.
- requirements-lock.txt records the actual complete environment; requirements.txt pins direct dependencies. Verified on Python 3.15.2/macOS arm64. Other Python/platform combinations were not tested.
- SE2418 group code inferred from the user's existing Midterm_Zholdubayev_Diyar_SE2418 archive/folder displayed in Finder, not explicitly confirmed. Archive naming uses Endterm_Zholdubayev_Diyar_SE2418.zip; verify this group before Moodle submission. Both students can upload the same full archive as the rubric specifies.
- Archive builder excludes virtual environments, clean-room builds, caches and PowerPoint owner files. It tests ZIP CRCs and every member's SHA-256 against SUBMISSION_MANIFEST.json.
- No Git commit or push was made.

## Remaining human requirements
Confirm group, review automated work, record verified contributions, rehearse both speaking sections, and practice independent explanations of all methods. Criterion 8 and actual seven-minute delivery cannot be verified by generated materials. Partner's new Endterm contribution remains unassigned. No guaranteed score is claimed.

## Working-tree cleanup requested after completion
The standalone Midterm folder and old presentation were removed from the working tree after
being verified and backed up to archive/midterm_originals.zip. This local backup is ignored
by Git and included in the submission ZIP when present. Dataset, SQLite and both saved ID
files moved without byte changes to data/raw/. The original baseline script moved with its executable logic unchanged
to src/_midterm_reference.py because historical reproduction still uses its pure functions.
The active SHA-256 inventory now covers these five canonical files; the initial full inventory
is inside the backup. All 11 tests passed after relocation. Temporary build/clean_room,
preview PNGs, PDF text extracts and Python caches were removed. .venv is retained for running
the project but excluded from Git, as are .idea, archive backups and submission ZIPs.
The notebook and clean-environment logs document the earlier execution paths; data/model
settings and numeric outputs are unchanged by relocation. The clean-room environment and
preview files named above were deliberately removed after verification.

## Comment cleanup
Removed Python comments and docstrings from src/ and tests/ at the user’s request. AST comparison verified identical executable logic after stripping docstrings. The reference baseline checksum was updated for this formatting-only change; its original file remains in the local Midterm backup. Raw CSV, SQLite and saved split checksums are unchanged. AI disclosure remains in CONTRIBUTIONS.md.

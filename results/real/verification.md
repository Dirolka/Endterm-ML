# Real-data verification

The source Parquet was downloaded at pinned Hugging Face revision b56a6054c10f1266a141e37c8df66c79ff2863af. SHA-256: 71dddaf0d9660f34ddf946953b67f54f1e7d60e22dcf9a4b8139b6378b18aab9.

The six-model employer-grouped experiment completed. Source rows: 141897; uniform benchmark sample: 12000; saved train/test rows: 9906/2094. There is no normalized employer or exact-description overlap. Five grouped training CV folds were checked for employer overlap. Title/experience/employer perturbation leaves preferred-model predictions unchanged.

`python -m src.verify_real_artifacts` independently reloaded all six fitted models, matched every held-out prediction against the logs, recalculated accuracy and macro F1, checked each confusion matrix, source hash, saved split IDs/order and employer disjointness. Predictions are reproduced using only the approved input columns. All checks passed; see artifact_verification.json.

The new notebook notebooks/Real_Djinni_Experience.ipynb ran all six code cells successfully in the IDE kernel. It inspects saved outputs and reruns artifact verification by default. Its optional full-training path imports and calls the same experiment function; the actual full training completed through src.real_experiments before notebook execution.

PowerPoint opened the new editable deck without a repair prompt and exported a local print-optimized 16-page PDF. PDFKit rendered all sixteen pages to images. Visual review of the complete contact sheet found no clipped text, overlaps or missing diagrams. Eight data charts, architecture diagrams and the confusion table remain native editable objects in PPTX. The PDF is a fixed visual copy. Notes contain the English defense script and evidence references.

Convergence warnings in MLP are disclosed: selected fit hit 180 iterations. CV selection optimism, minority-band weakness, near-duplicate/employer alias limitations and lack of temporal evaluation remain. Dataset-card MIT versus upstream repository Apache-2.0 license discrepancy is recorded, with both source notices retained. No significance or production-validity claim is made.

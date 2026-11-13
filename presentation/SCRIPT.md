# Real-data defense script

Approximate target: 9–11 minutes; rehearse with the actual slides. Speaking roles should reflect actual contributions.

## Slide 1: IT Vacancy Experience Classification

Good afternoon. We revised our project after the feedback about synthetic data. This version uses actual job postings from the published Djinni Recruitment Dataset. The task is to predict an employer-declared experience band from indirect vacancy features. We compare six classifiers, select the model using employer-grouped training cross-validation, and report results on companies excluded from training. We call the classes experience bands rather than claiming independently verified Junior, Middle and Senior labels.

## Slide 2: Dataset provenance

The source is the English job-description portion of the Djinni Recruitment Dataset, published by Drushchak and Romanyshyn at UNLP in 2024. The downloaded file contains 141,897 rows. We pin the repository revision and record the file hash. After removing exact repeated descriptions, 141,897 records remain. We uniformly sample twelve thousand to bound training time. The vacancies are real; anonymized UUIDs identify corpus records and are not evidence of generated job content. No candidate CVs are used.

## Slide 3: Target: declared experience requirements

The original target is the employer-provided Exp Years field. We map no experience and one year to the first band, two and three years to the second band, and five years to the third band. These are explicit operational categories, not a claim that five years always means Senior. Unlike the previous experiment, no title-based rule creates labels. We exclude the source target column, title and company identity from model inputs. The chart shows the class distribution in the actual sampled benchmark.

## Slide 4: Vacancy categories

The corpus covers different technology and job categories. This chart shows the ten most frequent categories in our real sample. Category is an employer-provided input, not the experience label. We also use declared English level. We do not have reliable structured salary fields in this source, so we remove the salary pipeline rather than invent salaries or carry over synthetic values. These changes mean the new scores must be interpreted within this new benchmark and cannot be compared directly with old synthetic scores.

## Slide 5: Declared English requirements

The English-level field is a real categorical requirement in this source. Missing values are explicitly represented as Unknown rather than guessed. The chart describes the sample, not a causal relationship between language and experience. One-hot encoding learns the available categories within each training fold and ignores unseen categories at validation time. This field may reflect regional and employer preferences, so its predictive usefulness should not be interpreted as a universal measure of professional seniority.

## Slide 6: Employer-grouped evaluation architecture

We first remove exact repeated descriptions. A fixed group split produces 9,906 training rows from 2,811 company groups and 2,094 test rows from 703 groups. No normalized company name appears on both sides. Five-fold cross-validation also keeps employer groups apart. The preferred model is fixed by mean CV macro F1 before test predictions. This is stricter than a random row split, but company-name normalization cannot perfectly resolve corporate aliases, and near-duplicate templates may remain.

## Slide 7: Preprocessing and feature architecture

Our model input contains twenty-nine numeric features: description length, word count, technology count and twenty-six fixed technology indicators. We detect technology names with word boundaries, avoiding the old Go-inside-Django substring problem. Primary job category and English level are one-hot encoded. Imputation, scaling and category vocabularies are fitted inside each CV fold. Raw title, full text, company identity and the source experience target are excluded. Raw text still influences length and technology counts, so these features are indirect signals rather than proof of semantic understanding.

## Slide 8: Six-model comparison

The chart compares mean employer-grouped training CV macro F1 with test macro F1. Macro F1 gives equal weight to each experience band. The preferred model is MLP, with mean CV F1 0.4632 and fold standard deviation 0.0152. Its held-out F1 is 0.4776. The standard deviation describes variation across folds and is not a confidence interval. We do not claim statistical significance or a globally optimal parameter configuration. These real-data results replace the synthetic numbers for the current defense.

## Slide 9: Ensemble architectures

Random Forest trains randomized trees on bootstrap samples and combines their votes. Our grid varies maximum depth and minimum leaf size, with one hundred and fifty trees fixed. Histogram Gradient Boosting instead adds trees sequentially to reduce the current loss. Its grid varies leaf complexity and L2 regularization, with one hundred and fifty iterations fixed. Each model has four configurations evaluated on the same five grouped folds. The smaller grids bound computation; selection still uses training data only.

## Slide 10: MLP: confusion matrix

This matrix uses only held-out employers. Rows are actual experience bands and columns are predictions. Diagonal entries are correct predictions. The preferred model makes 794 errors among 2,094 records. Off-diagonal counts show which requirements are confused. These are mistakes against employer-declared categories, not against an independently audited measure of professional competence. Every prediction is saved with the original corpus ID for inspection.

## Slide 11: Performance by experience band

This chart separates precision, recall and F1 for each band. Precision asks how often a prediction of the band is correct. Recall asks how many actual members of that band are found. A strong overall accuracy can hide a weak class, which is why macro F1 is primary. Class difficulty must be read from the new real-data report; we cannot reuse the old statement about the hardest synthetic class.

## Slide 12: MLP architecture

The neural network uses 76 encoded inputs, one hidden layer with thirty-two ReLU units and a three-class softmax output. Adam optimizes the weights. Two L2 regularization values are compared using grouped CV. The final fit performs 180 iterations out of a maximum of 180. Reaching the cap means convergence is not established. A separate grouped training-only diagnostic tracks learning without looking at the final test set.

## Slide 13: MLP learning progress

The diagnostic holds out employer groups from the training set: 7752 fitting rows and 2154 validation rows. We fit preprocessing only on the fitting partition and record F1 over one hundred epochs. Final diagnostic training F1 is 0.548 and validation F1 is 0.439. The curves help assess generalization and optimization, but this diagnostic network is separate from the final full-training fit. No test-guided stopping is used.

## Slide 14: PCA projection of training vacancies

PCA is fitted only on encoded training features and does not use experience labels. The first two components retain 20.8% of variance. The scatter colors are added afterward for interpretation. Overlap does not imply that the task is impossible: the projection discards information. Apparent separation also does not prove natural seniority groups. PCA is an explanatory diagnostic and is not part of the six-model classification pipelines.

## Slide 15: PCA explained variance

The cumulative curve quantifies information retained by the projection. In this benchmark, 27 components are needed to reach ninety-five percent of variance out of 76 encoded features. The numerical feature block is standardized within training; categorical indicators are one-hot encoded. This is a different representation from the old synthetic experiment. Explained variance is not classification accuracy, so we do not interpret ninety-five percent retained variance as ninety-five percent correct predictions.

## Slide 16: Conclusions and remaining limitations

The current result is a real-data benchmark with MLP preferred by grouped training CV and test macro F1 0.4776. We keep the synthetic experiment as historical evidence but do not compare its scores directly with the new task. The labels describe employer requirements rather than verified ability, the sample is limited to English Djinni postings, and near-duplicate text or employer aliases may remain. The next steps are a larger sample, stronger duplicate grouping and a chronological external evaluation. We will report improvements or failures using the same fixed evaluation protocol. Thank you.
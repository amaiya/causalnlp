# Changes

Most recent releases are shown at the top. Each release shows:

- **New**: New classes, methods, functions, etc
- **Changed**: Additional parameters, changes to inputs or outputs, etc
- **Fixed**: Bug fixes that don't change documented behaviour

## 0.7.0 (2022-08-02)

### New:
- N/A

### Changed
- updated dependencies

### Fixed:
- N/A


## 0.6.0 (2021-10-20)

### New:
- Added `model_name` parameter to `CausalBertModel` to support other DistilBert models (e.g., multilingual)

### Changed
- N/A

### Fixed:
- N/A


## 0.5.0 (2021-09-03)

### New:
- Added support for `CausalBert`

### Changed
- Added `p` parameter to `CausalInferenceModel.fit` and `CausalInferenceModel.predict` for user-supplied propensity scores in X-Learner and R-Learner.
- Removed CV from propensity score computations in X-Learner and R-Learner and increase default `max_iter` to 10000

### Fixed:
- Resolved problem with `CausalInferenceModel.tune_and_use_default_learner` when outcome is continuous
- Changed to `max_iter=10000` for default `LogisticRegression` base learner


## 0.4.0 (2021-07-20)

### New:
- N/A

### Changed
- Use `LinearRegression` and `LogisticRegression` as default base learners for `s-learner`.
- changed parameter name of `metalearner_type` to `method` in `CausalInferenceModel`.

### Fixed:
- Resolved mis-references in `_balance` method (renamed from `_minimize_bias`).
- Fixed convergence issues and factored out propensity score computations to `CausalInferenceModel.compute_propensity_scores`.


## 0.3.1 (2021-07-19)

### New:
- N/A

### Changed
- N/A

### Fixed:
- Added `sample_size` parameter to `CausalInferenceModel.evalute_robustness`


## 0.3.0 (2021-07-15)

### New:
- Added `CausalInferenceModel.evaluate_robustness` method to assess robustness of causal estimates using sensitivity analysis

### Changed
- reduced dependencies with local metalearner implementations

### Fixed:
- N/A


## 0.2.0 (2021-06-21)

### New:
- key driver analysis

### Changed
- `CausalInfererenceModel.fit` returns  `self`

### Fixed:
- N/A

## 0.1.3 (2021-06-17)

### New:
- N/A

### Changed
- N/A

### Fixed:
- version fix


## 0.1.2 (2021-06-17)

### New:
- N/A

### Changed
- Better interpretability and explainability of treatment effects

### Fixed:
- Fixes to some bugs in preprocessing


## 0.1.1 (2021-06-16)

### New:
- N/A

### Changed
- Refactored DataFrame preprocessing

### Fixed:
- N/A



## 0.1.0 (2021-06-15)

### New:
- First release.

### Changed
- N/A

### Fixed:
- N/A




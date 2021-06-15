# CausalNLP
> **CausalNLP** is a practical toolkit for causal inference with **text**


## Install

`pip install causalnlp`

## Usage

### What is the causal impact of a positive review on a product click?

```python
import pandas as pd
df = pd.read_csv('sample_data/music_seed50.tsv', sep='\t', error_bad_lines=False)
```

The file `music_seed50.tsv` is a semi-simulated dataset from [here](https://github.com/rpryzant/causal-text). Columns of relevance include:
- `C_true`:known confounding catgorical variable (1=audio CD, 0=other)
- `Y_sim`: simulated outcome, where 1 means product was clicked and 0 means not. 
- `T_true`: 1 means rating less than 3, 0 means rating of 5, where `T_true` affects the outcome `Y_sim`.
- `T_ac`: An approximation of true review sentiment (`T_true`) created with `Autocoder`.

We'll pretend the rating and `T_true` are unobserved and only use `T_ac` as the treatment variable. Using the `text_col` parameter, we include raw text as covariates for which adjustments can be made to improve causal estimates.

```python
from causalnlp.causalinference import CausalInferenceModel
from lightgbm import LGBMClassifier
cm = CausalInferenceModel(df, 
                         metalearner_type='t-learner', learner=LGBMClassifier(num_leaves=500),
                         treatment_col='T_ac', outcome_col='Y_sim', text_col='text',
                         include_cols=['C_true'])
cm.fit()
```

    outcome column (categorical): Y_sim
    treatment column: T_ac
    numerical/categorical covariates: ['C_true']
    text covariate: text
    preprocess time:  1.1460130214691162  sec
    start fitting causal inference model
    time to fit causal inference model:  15.419760704040527  sec


The average treatment effect (ATE):

```python
cm.estimate_ate()
```




    {'ate': 0.1309311542209525}



The conditional average treatment effect (CATE) for those reviews that mention the word "toddler":

```python
cm.estimate_ate(df['text'].str.contains('toddler'))
```




    {'ate': 0.15559234254638685}



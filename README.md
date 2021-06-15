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

```python
print(df.head(2).to_markdown())
```

    |    |   index |         id |   rating | product   | text                                                                                                                                     | summary                          |   price |   T_true |   C_true |   Y_sim |   negative |   positive |   T_ac |
    |---:|--------:|-----------:|---------:|:----------|:-----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|--------:|---------:|---------:|--------:|-----------:|-----------:|-------:|
    |  0 |       7 | 0001388703 |        1 | mp3 music | buy the cd.  do not buy the mp3 album.  download is no longer available.  but you don't find that out until after you have purchased it. | Buy the CD.  Do not buy the MP3. |   13.01 |        0 |        0 |       0 | 0.548733   |   0.451267 |      0 |
    |  1 |       8 | 0001388703 |        5 | mp3 music | takes me back to my childhood!                                                                                                           | Love it!                         |   13.01 |        1 |        0 |       0 | 0.00837317 |   0.991627 |      1 |


This is a semi-simulated dataset from [here](https://github.com/rpryzant/causal-text).
- `C_true`:known confounding catgorical variable (1=audio CD, 0=other)
- `Y_sim`: simulated outcome, where 1 means product was clicked and 0 means not. 
- `T_true`: 1 means rating less than 3, 0 means rating of 5.
- `T_ac`: An approximation of true review sentiment (`T_true`) created with `causalnlp.autocoder.Autocoder`.

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
    preprocess time:  1.1067862510681152  sec
    start fitting causal inference model
    time to fit causal inference model:  10.546765327453613  sec


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



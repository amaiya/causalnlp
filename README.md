# CausalNLP
> CausalNLP is a practical toolkit for causal inference with text as treatment, outcome, or "controlled-for" variable.


## Install

1. `pip install -U pip`
2. `pip install causalnlp`

## Usage

### Example: What is the causal impact of a positive review on a product click?

```python
import pandas as pd
df = pd.read_csv('sample_data/music_seed50.tsv', sep='\t', error_bad_lines=False)
```

The file `music_seed50.tsv` is a semi-simulated dataset from [here](https://github.com/rpryzant/causal-text). Columns of relevance include:
- `Y_sim`: outcome, where 1 means product was clicked and 0 means not.
- `text`: raw text of review
- `rating`: rating associated with review (1 through 5)
- `T_true`: 1 means rating less than 3, 0 means rating of 5, where `T_true` affects the outcome `Y_sim`.
- `T_ac`: an approximation of true review sentiment (`T_true`) created with [Autocoder](https://amaiya.github.io/causalnlp/autocoder.html) from raw review text
- `C_true`:confounding categorical variable (1=audio CD, 0=other)


We'll pretend the true sentiment (i.e., review rating and `T_true`) is hidden and only use `T_ac` as the treatment variable. 

Using the `text_col` parameter, we include the raw review text as another "controlled-for" variable.

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
    preprocess time:  1.1179866790771484  sec
    start fitting causal inference model
    time to fit causal inference model:  10.361494302749634  sec


#### Estimating Treatment Effects

CausalNLP supports estimation of heterogeneous treatment effects (i.e., how causal impacts vary across observations, which could be documents, emails, posts, individuals, or organizations).

We will first calculate the overall average treatment effect (or ATE), which shows that a positive review increases the probability of a click by **13 percentage points** in this dataset.

**Average Treatment Effect** (or **ATE**):

```python
print( cm.estimate_ate() )
```

    {'ate': 0.1309311542209525}


**Conditional Average Treatment Effect** (or **CATE**): reviews that mention the word "toddler":

```python
print( cm.estimate_ate(df['text'].str.contains('toddler')) )
```

    {'ate': 0.15559234254638685}


 **Individualized Treatment Effects** (or **ITE**):

```python
test_df = pd.DataFrame({'T_ac' : [1], 'C_true' : [1], 
                        'text' : ['I never bought this album, but I love his music and will soon!']})
effect = cm.predict(test_df)
print(effect)
```

    [[0.80538201]]


**Model Interpretability**:

```python
print( cm.interpret(plot=False)[1][:10] )
```

    v_music    0.079042
    v_cd       0.066838
    v_album    0.055168
    v_like     0.040784
    v_love     0.040635
    C_true     0.039949
    v_just     0.035671
    v_song     0.035362
    v_great    0.029918
    v_heard    0.028373
    dtype: float64


Features with the `v_` prefix are word features. `C_true` is the categorical variable indicating whether or not the product is a CD. 

## Documentation
API documentation and additional usage examples are available at: https://amaiya.github.io/causalnlp/

## How to Cite

Please cite [the following paper](https://arxiv.org/abs/2106.08043) when using CausalNLP in your work:

```
@article{maiya2021causalnlp,
    title={CausalNLP: A Practical Toolkit for Causal Inference with Text},
    author={Arun S. Maiya},
    year={2021},
    eprint={2106.08043},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    journal={arXiv preprint arXiv:2106.08043},
}
```

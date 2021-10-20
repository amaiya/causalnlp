# Welcome to CausalNLP



## What is CausalNLP?
> CausalNLP is a practical toolkit for causal inference with text as treatment, outcome, or "controlled-for" variable.

## Features
- Low-code [causal inference](https://amaiya.github.io/causalnlp/examples.html) in as little as two commands
- Out-of-the-box support for using [**text** as a "controlled-for" variable](https://amaiya.github.io/causalnlp/examples.html#What-is-the-causal-impact-of-a-positive-review-on-product-views?) (e.g., confounder)
- Built-in [Autocoder](https://amaiya.github.io/causalnlp/autocoder.html) that transforms raw text into useful variables for causal analyses (e.g., topics, sentiment, emotion, etc.)
- Sensitivity analysis to [assess robustness of causal estimates](https://amaiya.github.io/causalnlp/causalinference.html#CausalInferenceModel.evaluate_robustness)
- Quick and simple [key driver analysis](https://amaiya.github.io/causalnlp/key_driver_analysis.html) to yield clues on potential drivers of an outcome based on predictive power, correlations, etc.
- Can easily be applied to ["traditional" tabular datasets without text](https://amaiya.github.io/causalnlp/examples.html#What-is-the-causal-impact-of-having-a-PhD-on-making-over-$50K?) (i.e., datasets with only numerical and categorical variables)
- Includes an experimental [PyTorch implementation](https://amaiya.github.io/causalnlp/core.causalbert.html) of [CausalBert](https://arxiv.org/abs/1905.12741) by Veitch, Sridar, and Blei (based on [reference implementation](https://github.com/rpryzant/causal-bert-pytorch) by R. Pryzant)

## Install

1. `pip install -U pip`
2. `pip install causalnlp`

**NOTE**: On Python 3.6.x, if you get a `RuntimeError: Python version >= 3.7 required`, try ensuring NumPy is installed **before** CausalNLP (e.g., `pip install numpy==1.18.5`).

## Usage

To try out the [examples](https://amaiya.github.io/causalnlp/examples.html) yourself:

<a href="https://colab.research.google.com/drive/1hu7j2QCWkVlFsKbuereWWRDOBy1anMbQ?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Example: What is the causal impact of a positive review on a product click?

```
import pandas as pd
df = pd.read_csv('sample_data/music_seed50.tsv', sep='\t', error_bad_lines=False)
```

The file `music_seed50.tsv` is a semi-simulated dataset from [here](https://github.com/rpryzant/causal-text). Columns of relevance include:
- `Y_sim`: outcome, where 1 means product was clicked and 0 means not.
- `text`: raw text of review
- `rating`: rating associated with review (1 through 5)
- `T_true`: 0 means rating less than 3, 1 means rating of 5, where `T_true` affects the outcome `Y_sim`.
- `T_ac`: an approximation of true review sentiment (`T_true`) created with [Autocoder](https://amaiya.github.io/causalnlp/autocoder.html) from raw review text
- `C_true`:confounding categorical variable (1=audio CD, 0=other)


We'll pretend the true sentiment (i.e., review rating and `T_true`) is hidden and only use `T_ac` as the treatment variable. 

Using the `text_col` parameter, we include the raw review text as another "controlled-for" variable.

```
from causalnlp import CausalInferenceModel
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

```
print( cm.estimate_ate() )
```

    {'ate': 0.1309311542209525}


**Conditional Average Treatment Effect** (or **CATE**): reviews that mention the word "toddler":

```
print( cm.estimate_ate(df['text'].str.contains('toddler')) )
```

    {'ate': 0.15559234254638685}


 **Individualized Treatment Effects** (or **ITE**):

```
test_df = pd.DataFrame({'T_ac' : [1], 'C_true' : [1], 
                        'text' : ['I never bought this album, but I love his music and will soon!']})
effect = cm.predict(test_df)
print(effect)
```

    [[0.80538201]]


**Model Interpretability**:

```
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

### Text is Optional in CausalNLP

Despite the "NLP" in CausalNLP, the library can be used for causal inference on data **without** text (e.g., only numerical and categorical variables). See [the examples](https://amaiya.github.io/causalnlp/examples.html#What-is-the-causal-impact-of-having-a-PhD-on-making-over-$50K?) for more info.

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

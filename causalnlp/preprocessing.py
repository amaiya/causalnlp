# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_preprocessing.ipynb.

# %% auto 0
__all__ = ['DataframePreprocessor']

# %% ../nbs/04_preprocessing.ipynb 4
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import time


class DataframePreprocessor:
    """
    Preproceses a pandas DataFrame for causal inference
    """
    def __init__(self, 
                 treatment_col='treatment', 
                 outcome_col='outcome', 
                 text_col=None,
                 include_cols=[],
                 ignore_cols=[],
                 verbose=1):
        """
        Instantiates the DataframePreprocessor instance.
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.text_col = text_col
        self.include_cols = include_cols
        self.ignore_cols = ignore_cols
        self.v = verbose

        
        # these variables set by preprocess
        self.feature_names = None
        self.feature_names_one_hot = None
        self.feature_types = {}
        self.cat_dict = {}
        self.tv = None
        self.is_classification = None


    def preprocess(self, df, 
                   training=False,
                   min_df=0.05,
                   max_df=0.5,
                   ngram_range=(1,1),
                   stop_words='english',
                   na_cont_value=-1, na_cat_value='MISSING'):
        """
        Preprocess a dataframe for causal inference.
        """
        # checks
        if not training and self.feature_names is None:
            raise ValueError('Preprocessor must first be fitted by calling with training=True.')
        if not isinstance(self.ignore_cols, list):
            raise ValueError('ignore_cols must be a list.')
        if not isinstance(self.include_cols, list):
            raise ValueError('include_cols must be a list.')
        if training and self.ignore_cols and self.include_cols:
            raise  ValueError('ignore_cols and include_cols are mutually exclusive.  Please choose one.')
        if training and self.include_cols:
            self.ignore_cols = [c for c in df.columns.values if c not in self.include_cols +\
                                                                              [self.treatment_col, 
                                                                               self.outcome_col, 
                                                                               self.text_col]]
        if self.text_col is not None and self.text_col not in df:
            raise ValueError(f'You specified text_col="{self.text_col}", but {self.text_col} is not a column in df.')
        if self.treatment_col in self.ignore_cols:
            raise ValueError(f'ignore_cols contains the treatment column ({self.treatment_col})')
        if self.outcome_col in self.ignore_cols:
            raise ValueError(f'ignore_cols contains the outcome column ({self.outcome_col})')
            
        start_time = time.time()
        
        # step 1: check/clean dataframe
        if not isinstance(df, pd.DataFrame):
            raise ValueError('df must be a pandas DataFrame')
        df = df.rename(columns=lambda x: x.strip()) # strip headers 
        # check and re-order test DataFrame
        if not training:
            test_feats = [col.strip() for col in df.columns.values if col.strip() in self.feature_names]
            if len( set(test_feats) & set(self.feature_names) ) != len(self.feature_names):
                raise ValueError('df must contain the same columns as DataFrame used for training model.')
            if self.treatment_col not in df.columns:
                raise ValueError(f'Column {self.treatment_col} is missing from df.')
            if self.text_col is not None and self.text_col not in df.columns.values:
                raise ValueError(f'Colummn {self.text_col} is missing from df')               
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # strip data
        df, _ = self._preprocess_column(df, self.treatment_col, is_treatment=True)
        if training:
            df, self.is_classification = self._preprocess_column(df, 
                                                                 self.outcome_col, is_treatment=False)
            self.feature_names = [c for c in df.columns.values \
                                  if c not in [self.treatment_col, 
                                              self.outcome_col, self.text_col]+self.ignore_cols]
            for c in self.feature_names:
                self.feature_types[c] = self._check_type(df, c)['dtype']
        X = df[self.feature_names].copy()
        Y = df[self.outcome_col].copy() if training else None
        T = df[self.treatment_col].copy()   

        # step 2: fill empty values on x
        for c in self.feature_names:
            dtype = self.feature_types[c]  
            if dtype == 'string': X[c] = X[c].fillna(na_cat_value)
            if dtype == 'numeric': X[c] = X[c].fillna(na_cont_value)
                        
        # step 3: one-hot encode categorial features
        for c in self.feature_names:
            if c == self.text_col: continue
            if self.feature_types[c] == 'string':
                if df.shape[0] > 100 and df[c].nunique()/df.shape[0] > 0.5:
                    if self.text_col is not None:
                        err_msg = f'Column "{c}" looks like it contains free-form text. ' +\
                        f'Since there is already a text_col specified ({self.text_col}), '+\
                        f'you should probably include this column in the "ignore_cols" list.'
                    else:
                        err_msg = f'Column "{c}" looks like it contains free-form text or ' +\
                        f'or unique values. Please either set text_col="{c}" or add it to "ignore_cols" list.'
                    raise ValueError(err_msg)
                      
                if training:
                    self.cat_dict[c] = sorted(X[c].unique())
                    catcol = X[c]
                else:
                    #REF: https://stackoverflow.com/a/37451867/13550699
                    catcol = X[c].astype(pd.CategoricalDtype(categories=self.cat_dict[c]))
                X = X.merge(pd.get_dummies(catcol, prefix = c, 
                                                     drop_first=False), 
                                                     left_index=True, right_index=True)
                
                del X[c]
        self.feature_names_one_hot = X.columns
        
                        
        # step 4: for text-based confounder, use extracted vocabulary as features
        if self.text_col is not None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            if training:
                self.tv = TfidfVectorizer(min_df=min_df, max_df=max_df, 
                                         ngram_range=ngram_range, stop_words=stop_words)
                v_features = self.tv.fit_transform(df[self.text_col])
            else:
                v_features = self.tv.transform(df[self.text_col])
            vocab = self.tv.get_feature_names_out()
            vocab_df = pd.DataFrame(v_features.toarray(), columns = ["v_%s" % (v) for v in vocab])
            X = pd.concat([X, vocab_df], axis=1, join='inner')
        outcome_type = 'categorical' if self.is_classification else 'numerical'
        if training:
            if self.outcome_col in df.columns and self.v:
                print(f'outcome column ({outcome_type}): {self.outcome_col}')
            if self.v: print(f'treatment column: {self.treatment_col}')
            if self.v: print('numerical/categorical covariates: %s' % (self.feature_names))
            if self.v and self.text_col: print('text covariate: %s' % (self.text_col))
            if self.v: print("preprocess time: ", -start_time + time.time()," sec")
        return (df, X, Y, T)
        
        
    def _preprocess_column(self, df, col, is_treatment=True):
        """
        Preprocess treatment and outcome columns.
        """
        # remove nulls
        df = df[df[col].notnull()]

        # check if already binarized
        if self._check_binary(df, col): return df, True

        # inspect column
        d = self._check_type(df, col)
        typ = d['dtype']
        num = d['nunique']
        
        # process as treatment
        if is_treatment:
            if typ == 'numeric' or (typ == 'string' and num != 2): 
                raise ValueError('Treatment column must contain only two unique values ' +\
                                 'indicating the treated and control groups.')
            values = sorted(df[col].unique())
            df[col].replace(values, [0,1], inplace=True)
            if self.v: print('replaced %s in column "%s" with %s' % (values, col, [0,1]))
        # process as outcome
        else:
            if typ == 'string' and num != 2:
                raise ValueError('If the outcome column is string/categorical, it must '+
                                'contain only two unique values.')
            if typ == 'string':
                values = sorted(df[col].unique())
                df[col].replace(values, [0,1], inplace=True)
                if self.v: print('replaced %s in column "%s" with %s' % (values, col, [0,1]))
        return df, self._check_binary(df, col)
        
        
    def _check_type(self, df, col):
        from pandas.api.types import is_string_dtype
        from pandas.api.types import is_numeric_dtype
        dtype = None
        
        tmp_var = df[df[col].notnull()][col]
        if is_numeric_dtype(tmp_var): dtype = 'numeric'
        elif is_string_dtype(tmp_var): dtype =  'string'
        else:
            raise ValueError('Columns in dataframe must be either numeric or strings.  ' +\
                             'Column %s is neither' % (col))
        output = {'dtype' : dtype, 'nunique' : tmp_var.nunique()}
        return output
    

    def _check_binary(self, df, col):
        return df[col].isin([0,1]).all()        

    def _get_feature_names(self, df):
        return [c for c in df.columns.values \
                if c not in [self.treatment_col, self.outcome_col]+self.ignore_cols]   

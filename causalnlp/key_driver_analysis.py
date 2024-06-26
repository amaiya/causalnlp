# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_key_driver_analysis.ipynb.

# %% auto 0
__all__ = ['KeyDriverAnalysis']

# %% ../nbs/03_key_driver_analysis.ipynb 5
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt


from .preprocessing import DataframePreprocessor



class KeyDriverAnalysis:
    """
    Performs key driver analysis
    """
    def __init__(self, df, outcome_col='outcome', text_col=None, include_cols=[], ignore_cols=[],
                 verbose=1):
        """
        Instantiates the KeyDriverAnalysis instance.
        """
        self.v = verbose
        self.pp = None # set with call to _preprocess
        self.df, self.x, self.y = self._preprocess(df, outcome_col=outcome_col, text_col=text_col, 
                                                   include_cols=include_cols, ignore_cols=ignore_cols)
        

    def _preprocess(self, df, outcome_col='outcome', text_col=None, include_cols=[], ignore_cols=[]):
        """
        preprocesses DataFrame
        """
        temp_treatment = 'CausalNLP_temp_treatment'
        df = df.copy()
        df[temp_treatment] = [0] * df.shape[0]
        
        # preprocess
        self.pp = DataframePreprocessor(treatment_col = temp_treatment,
                                       outcome_col = outcome_col,
                                       text_col=text_col,
                                       include_cols=include_cols,
                                       ignore_cols=ignore_cols,
                                       verbose=self.v)
        df, x, y, _ = self.pp.preprocess(df,
                                         training=True,
                                         min_df=0.05,
                                         max_df=0.5,
                                         ngram_range=(1,1),
                                         stop_words='english')
        return df, x, y

    def correlations(self, outcome_only=True):
        """
        Computes corelations between independent variables and outcome
        """

        df = self.x.copy()
        df[self.pp.outcome_col] = self.y
        corrALL = df.apply(pd.to_numeric, errors='coerce').corr()
        if outcome_only:
            df_results = corrALL[[self.pp.outcome_col]]
            df_results = df_results.sort_values(by=self.pp.outcome_col, key=abs, ascending=False)
            return df_results.iloc[1: , :]

            #return df_results.sort_values(by=[self.pp.outcome_col])
        else:
            return corrALL

        
        
    def importances(self, plot=True, split_pct=0.2, 
                    use_shap=False, shap_background_size=50,
                    rf_model=None, n_estimators=100, n_jobs=-1, random_state=42):
        """
        Identifies important predictors using a RandomForest model.
        """
       
        X_train, X_test, y_train, y_test = train_test_split(self.x.values, self.y.values, 
                                                            test_size=split_pct, 
                                                            random_state=random_state)
        rf_type = RandomForestClassifier if self.pp.is_classification else RandomForestRegressor
        rf = rf_type(n_estimators = n_estimators,
                                   n_jobs = n_jobs,
                                   oob_score = True,
                                   bootstrap = True,
                                   random_state = random_state)
        rf.fit(X_train, y_train)
        if self.v:
            print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(
                                                                             rf.score(X_train, y_train), 
                                                                             rf.oob_score_,
                                                                             rf.score(X_test, y_test)))
        driverNames = self.x.columns.values
        if use_shap:
            try:
                import shap
            except ImportError:
                raise ImportError('Please install shap (conda recommended): '+\
                                 'conda install shap --channel conda-forge')
            explainer = shap.KernelExplainer(rf.predict, X_test[:shap_background_size,:])
            shap_values = explainer.shap_values(X_test[:shap_background_size,:])
            if plot:
                shap.summary_plot(shap_values, X_test[:shap_background_size,:], feature_names=driverNames)
            vals = np.abs(shap_values).mean(0)

            df_results = pd.DataFrame(list(zip(driverNames, vals)),
                                  columns=['Driver','Importance'])
            df_results.sort_values(by=['Importance'],
                                   ascending=False, inplace=True)
            return df_results
        else:
            df_results = pd.DataFrame(data = {'Driver': driverNames,
                                             'Importance': rf.feature_importances_})
            df_results = df_results.sort_values('Importance', ascending=False)
            if plot:
                feat_importances = pd.Series(rf.feature_importances_, index=driverNames)
                feat_importances.nlargest(20).plot(kind='barh')
            return df_results        

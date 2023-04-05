import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression, Ridge, ElasticNet, Lasso,
                                  TheilSenRegressor, RANSACRegressor,
                                  HuberRegressor, SGDRegressor, Lars,
                                  RidgeCV)
from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:
    """
    Class used for iterating through two dictionaries - one with list of
    models, and one with list of parameters (describing hyperparameter space).
    Via source materials: "The code above defines the helper class, now you
    need to pass it a dictionary of models and a dictionary of parameters for
    each of the models."

    Library imports:
    from sklearn.model_selection import GridSearchCV

    Source materials:
    1. https://www.davidsbatista.net/blog/2018/02/23/model_optimization/

    Use k-fold linear regression to estimate average score per #parameters.

    Estimate average R-squared adjustment on kfold cross-validated sample
    grouped by number of explanatory variables used. Applicable for regression
    tasks. Using cross-validation tests average. Estimated by linear model.

    Library imports:
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression

    Parameters
    ----------
    data_in : str
        DataFrame with independent variables to analyze.
    features_in : str
        List of variables to be chosen from. Must come from data_in DataFrame.
    target : str
        Series with dependent variable. Must be continuous.
    random_state : int, default = 123
        Random number generator seed. used for KFold sampling.
    max_features : int, default = 10
        Limit of features in iteration. Algorithm will compute for models from
        i = 1 feature to max_features.
    n_splits : int, default = 5
        Cross-validation parameter - will split data_in to n_splits equal
        parts. VarClusHi library).
    shuffle : bool, default = True
        Whether to shuffle the data before splitting into batches. Note that
        the samples within each split will not be shuffled.

    Returns
    -------
    table: top10 scores
        Top 10 R-squared scores by mean test-set score and corresponding number of
        features category.
    plot: mean scores plot
        Line plot of number of features selected versus average train & test
        sample R-squared scores.

    Notes
    -------------------
    Required libraries: \n
    import pandas as pd \n
    import numpy as np \n
    from sklearn.linear_model import LinearRegression \n
    from sklearn.linear_model import Ridge \n
    from sklearn.linear_model import ElasticNet \n
    from sklearn.linear_model import Lasso \n
    from sklearn.linear_model import TheilSenRegressor \n
    from sklearn.linear_model import RANSACRegressor \n
    from sklearn.linear_model import HuberRegressor \n
    from sklearn.linear_model import SGDRegressor \n
    from sklearn.linear_model import Lars \n
    from sklearn.linear_model import RidgeCV \n
    from sklearn.model_selection import GridSearchCV
    """
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s"
                             % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X, y)
            self.grid_searches[key] = gs

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score',
                   'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

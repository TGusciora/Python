import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (KFold, GridSearchCV)
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


def kfold_nFeaturesSelector(data_in, features_in, target, random_state,
                            max_features, n_splits=5, shuffle=True):
    """
    Uses k-fold linear regression to estimate average score per #parameters.

    Applicable for regression tasks. Using cross-validation tests average
    R-Squared metric on validation dataset based on number of chosen
    variables. Estimated by linear model.

    Library imports:
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression

    Parameters
    ----------
    data_in : str
        DataFrame with independent variables to analyze.
    target : str
        Series with dependent variable. Must be continuous.
    features_in : str
        List of variables to be chosen from. Must come from data_in DataFrame.
    max_features : int, default = 10
        Limit of features in iteration. Algorithm will compute for models from
        i = 1 feature to max_features.
    n_splits : int, default = 5
        Cross-validation parameter - will split data_in to n_splits equal
        parts. VarClusHi library)
    shuffle : bool, default = True
        Whether to shuffle the data before splitting into batches. Note that
        the samples within each split will not be shuffled.

    Returns
    -------

    Top 10 R-squared scores by mean test-set score and corresponding number of
    features category.
    Line plot of number of features selected versus average train & test
    sample R-squared scores.

    Notes
    -------------------
    Required libraries: \n
    import pandas as pd: \n
    import matplotlib.pyplot as plt: \n
    import numpy as np: \n
    from sklearn.model_selection import (KFold, GridSearchCV): \n
    from sklearn.linear_model import LinearRegression: \n
    from sklearn.feature_selection import RFE: \n

    References
    ----------
    Source materials: \n
    1. Binning library <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html> \n
    2. WOE <https://www.kaggle.com/code/jnikhilsai/cross-validation-with-linear-regression>
    """
    # cross-validation configuration
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Removing low variance to avoid problems with cross_validation
    features_exclude = list(data_in[features_in].std()[data_in[features_in].std() < 0.1].index)
    features_fin = list(set(features_in)-set(features_exclude))

    # hyperparameters configuration
    hyper_params = [{'n_features_to_select': list(range(1, max_features))}]

    # grid search
    # Model specification
    # Further improvement: handling multiple types of models
    lm = LinearRegression()
    lm.fit(data_in[features_fin], target)  # y_train_valid
    rfe = RFE(lm)

    # executing GridSearchCV()
    model_cv = GridSearchCV(estimator=rfe,
                            param_grid=hyper_params,
                            scoring='r2',
                            cv=folds,
                            verbose=1,
                            return_train_score=True)
    # fitting model on train_valid sample
    model_cv.fit(data_in[features_fin], target)
    # cv results
    cv_results = pd.DataFrame(model_cv.cv_results_)

    # print(cv_results)
    # plotting cv results
    plt.figure(figsize=(16, 6))
    plt.plot(cv_results["param_n_features_to_select"],
             cv_results["mean_test_score"])
    plt.plot(cv_results["param_n_features_to_select"],
             cv_results["mean_train_score"])
    plt.xlabel('number of features')
    plt.ylabel('R^2')
    plt.xticks(np.arange(1, max_features, 1.0))
    plt.ylim(ymax=1.0, ymin=0.5)
    plt.yticks(np.arange(0.5, 1, 0.05))
    plt.title("Optimal Number of Features")
    plt.legend(['valid score', 'train score'], loc='upper left')
    print(cv_results[["param_n_features_to_select", "mean_train_score",
                      "mean_test_score"]].sort_values("mean_test_score",
                                                      ascending=False).head(10))

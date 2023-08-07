import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def make_mi_scores(X, y):
    """
    Calculate Mutual Information (MI) scores for the features in the dataset.

    Parameters:
        X (pandas.DataFrame): The feature matrix.
        y (pandas.Series): The target variable.

    Returns:
        pandas.Series: A Series containing MI scores for each feature.
    """
    X = X.copy()
    categorical_feats = X.select_dtypes(include=['object', 'string'])

    for colname in categorical_feats:
        X[colname], _ = X[colname].factorize()

    discrete_features = [col in categorical_feats for col in X.columns]
    mi_scores_series = pd.Series(mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0),
                                 name="MI_scores", index=X.columns
                        )
    mi_scores_series.sort_values(ascending=False, inplace=True)

    return mi_scores_series


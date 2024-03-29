# preprocessing
import pandas as pd

def preprocess_data():
    """read csv file, drop irrelevant columns and all rows that contain NaN, convert all values into int/float, build
    np arrays of target and predictors"""
    df = pd.read_csv('titanic.csv')
    df = df.drop(['name', 'parch', 'sibsp', 'home.dest', 'boat', 'body', 'cabin', 'ticket'], axis = 1)
    df = df.dropna()
    df.replace(to_replace=['male', 'female'], value=(0, 1), inplace=True)
    df.replace(to_replace=['C', 'S', 'Q'], value=(0, 1, 2), inplace=True)
    y = df['survived'].values
    X = df.drop('survived', axis=1).values
    assert len(X) == len(y)
    return df, y, X

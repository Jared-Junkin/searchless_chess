from sklearn.datasets import load_iris
import numpy as np
from scipy.stats import zscore
import pandas as pd
def doPCA(df: pd.DataFrame, n: int=2)->None:
    # z-score standardize the data column-wise
    
    # calculate covariance matrix
    
    # perform eigendecomposition on covariance matrix to get eigenvectors and eigenvalues

    # sort eigenvectors in descending order

    # return the top n eigen0vectors iwth eigenvalues

    return 0
if __name__ == "__main__":
    # iris = load_iris()
    # df = pd.DataFrame(iris.data[:50], columns=iris.feature_names)
    # print(df.columns)
    # res = doPCA(df)
    # print(res)
    # # print(df['target'][:50])
    hashmap = {'a': 10, 'b': 5, 'c': 20, 'd':-1}
    print(hashmap.items())
    
    print(sorted(hashmap.items(), key=lambda x: x[1], reverse=True))
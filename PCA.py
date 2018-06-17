import numpy as np
import pandas as pd




def mean_normalization(X):
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    X_normalization = (X - mean) / stdev

    return X_normalization

def pca(X):
    M, N = X.shape

    Sigma = (1 / M) * X.dot(X.T)
    U, S, V = np.linalg.svd(Sigma)
    return U,S


dataFrame = pd.read_csv('Iris.csv')
data = dataFrame.values

X = data[:, 1:-1].astype(float)

X = mean_normalization(X)

U,S = pca(X)


S_appro = np.zeros(U.shape)

for i in range(len(S)):
    S_appro[i][i] = S[i]



k_list = [1,2,3]
for _ in k_list:
    sum1 = 0
    sum2 = 0
    for i in range(S.shape[0]):
        sum1 += S_appro[i][i]

    for i in range(_):
        sum2 += S_appro[i][i]


    print()
    print("num of K : ", _)
    print("Variance retained : ", round(sum2/sum1,2))



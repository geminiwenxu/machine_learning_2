import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

if __name__ == '__main__':
    df = pd.read_csv("/data/data_ex5", delimiter=',', header=None,
                     names=['d1', 'd2', 'd3'])
    print(df.tail())

    X_std = StandardScaler().fit_transform(df)
    print("standardized df: ", X_std)
    print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))
    cov_mat = np.cov(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)

    for ev in eig_vecs.T:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])

    # Explained Variance
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    # with plt.style.context('seaborn-whitegrid'):
    #     plt.figure(figsize=(6, 4))
    #
    #     plt.bar(range(3), var_exp, alpha=0.5, align='center',
    #             label='individual explained variance')
    #     plt.step(range(3), cum_var_exp, where='mid',
    #              label='cumulative explained variance')
    #     plt.ylabel('Explained variance ratio')
    #     plt.xlabel('Principal components')
    #     plt.legend(loc='best')
    #     plt.tight_layout()
    # plt.show()
    print(eig_pairs)
    print(eig_pairs[0][1])
    # Projection Matrix
    matrix_w = np.hstack((eig_pairs[1][1].reshape(3, 1),
                          eig_pairs[2][1].reshape(3, 1)))

    print('Matrix W:\n', matrix_w)
    Y = X_std.dot(matrix_w)
    print((Y))
    for i in range(150):
        plt.scatter(Y[i][0], Y[i][1], c='blue')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 3')
    plt.show()

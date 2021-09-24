import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def preprocess(df):
    mean_d1, mean_d2, mean_d3 = df.sum(axis=0) / 150
    temp = 0
    for index, row in df.iterrows():
        temp += (row['d1'] - mean_d1) ** 2 + (row['d2'] - mean_d2) ** 2 + (row['d3'] - mean_d3) ** 2
    sd = np.sqrt(temp / 149)
    for index, row in df.iterrows():
        df.at[index, 'd1'] = (row['d1'] - mean_d1) ** 2 / sd
        df.at[index, 'd2'] = (row['d2'] - mean_d2) ** 2 / sd
        df.at[index, 'd3'] = (row['d3'] - mean_d3) ** 2 / sd
    return df, mean_d1, mean_d2, mean_d3


def cov(new_df, mean_d1, mean_d2, mean_d3):
    for index, row in df.iterrows():
        new_df.at[index, 'd1'] = (row['d1'] - mean_d1)
        new_df.at[index, 'd2'] = (row['d2'] - mean_d2)
        new_df.at[index, 'd3'] = (row['d3'] - mean_d3)
    X = df.to_numpy()
    cov_matrix = X.T.dot(X) / 149
    return X, cov_matrix


def eigen(cov_matrix):
    eigen_value, eigen_vector = LA.eig(cov_matrix)
    return eigen_value, eigen_vector


def project_matrix(eigen_value, eigen_vector):
    eig_pairs = [(np.abs(eigen_value[i]), eigen_vector[:, i]) for i in range(len(eigen_value))]
    matrix_1 = np.hstack((eig_pairs[0][1].reshape(3, 1),
                          eig_pairs[1][1].reshape(3, 1)))

    matrix_2 = np.hstack((eig_pairs[1][1].reshape(3, 1),
                          eig_pairs[2][1].reshape(3, 1)))

    matrix_3 = np.hstack((eig_pairs[0][1].reshape(3, 1),
                          eig_pairs[2][1].reshape(3, 1)))

    matrix_4 = np.hstack((eig_pairs[0][1].reshape(3, 1),
                          eig_pairs[1][1].reshape(3, 1),
                          eig_pairs[2][1].reshape(3, 1)))
    print("eigenpairs: ", eig_pairs)
    return matrix_1, matrix_2, matrix_3, matrix_4


if __name__ == '__main__':
    df = pd.read_csv("/data/data_ex5", delimiter=',', header=None,
                     names=['d1', 'd2', 'd3'])

    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    # ax.scatter3D(df['d1'], df['d2'], df['d3'])
    # ax.set_xlabel('X-axis', fontweight='bold')
    # ax.set_ylabel('Y-axis', fontweight='bold')
    # ax.set_zlabel('Z-axis', fontweight='bold')
    # plt.title("simple 3D scatter plot")
    # plt.show()

    # new_df, mean_d1, mean_d2, mean_d3 = preprocess(df)
    # X, cov_matrix = cov(try_df, mean_d1, mean_d2, mean_d3)

    X_std = StandardScaler().fit_transform(df)

    cov_mat = np.cov(X_std.T)
    eigen_value, eigen_vector = eigen(cov_mat)
    print("eigen_value", eigen_value)
    print("eigen_vector", eigen_vector)

    variation_1 = eigen_value[0] / 149
    variation_2 = eigen_value[1] / 149
    variation_3 = eigen_value[2] / 149
    total_variation = variation_1 + variation_2 + variation_3
    # print(variation_1 / total_variation, variation_2 / total_variation, variation_3 / total_variation)

    matrix_1, matrix_2, matrix_3, matrix_4 = project_matrix(eigen_value, eigen_vector)

    # plot
    Y = X_std.dot(matrix_3)
    print("pc1 and pc3", matrix_3)
    # print(Y)
    for i in range(len(Y)):
        plt.scatter(Y[i][0], Y[i][1], c='blue')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 3')
    # plt.show()

    # Y = X_std.dot(matrix_4)
    # fig = plt.figure(figsize=(10, 7))
    # ax = plt.axes(projection="3d")
    # for i in range(150):
    #     ax.scatter3D(Y[i][0], Y[i][1], Y[i][2], c='blue')
    # ax.set_xlabel('Principal Component 1', fontweight='bold')
    # ax.set_ylabel('Principal Component 2', fontweight='bold')
    # ax.set_zlabel('Principal Component 3', fontweight='bold')
    # plt.title("simple 3D scatter plot")
    # plt.show()

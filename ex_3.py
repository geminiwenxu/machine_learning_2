import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances_argmin


def e_step(df, var_1, var_2, pie_1, pie_2):
    lst_res_1 = []
    lst_res_2 = []
    for index, row in df.iterrows():
        density_1 = var_1.pdf([row['d1'], row['d2']])
        density_2 = var_2.pdf([row['d1'], row['d2']])
        p_x = pie_1 * density_1 + pie_2 * density_2
        res_1 = np.float32((pie_1 * density_1) / p_x).item()
        lst_res_1.append(res_1)
        res_2 = np.float32((pie_2 * density_2) / p_x).item()
        lst_res_2.append(res_2)
        plt.scatter(df['d1'], df['d2'], c=lst_res_1, cmap='Greens')
    return lst_res_1, lst_res_2


def m_step(df, lst_res_1, lst_res_2, N_1, N_2):
    temp_1 = 0
    temp_2 = 0
    for index, row in df.iterrows():
        data = [np.float32(row['d1']).item(), np.float32(row['d2']).item()]
        temp_1 += lst_res_1[index] * np.array(data)
        temp_2 += lst_res_2[index] * np.array(data)

    mu_1_new = temp_1 / N_1
    mu_2_new = temp_2 / N_2

    temp_3 = 0
    temp_4 = 0
    for index, row in df.iterrows():
        data = [np.float32(row['d1']).item(), np.float32(row['d2']).item()]
        temp_3 += lst_res_1[index] * np.multiply((np.matrix(data) - mu_1_new), (np.matrix(data) - mu_1_new).T)
        temp_4 += lst_res_1[index] * np.multiply((np.matrix(data) - mu_2_new), (np.matrix(data) - mu_2_new).T)
    sigma_1_new = temp_3 / N_1
    sigma_2_new = temp_4 / N_2

    pie_1_new = N_1 / 100
    pie_2_new = N_2 / 100
    return mu_1_new, mu_2_new, sigma_1_new, sigma_2_new, pie_1_new, pie_2_new


def eva(mu_1_new, mu_2_new, sigma_1_new, sigma_2_new, pie_1_new, pie_2_new):
    var_1_new = multivariate_normal(mean=mu_1_new, cov=sigma_1_new)
    var_2_new = multivariate_normal(mean=mu_2_new, cov=sigma_2_new)
    likelihood = 0
    log_likelihood = 0
    for index, row in df.iterrows():
        density_1_new = var_1_new.pdf([row['d1'], row['d2']])
        density_2_new = var_2_new.pdf([row['d1'], row['d2']])
        likelihood = pie_1_new * density_1_new + pie_2_new * density_2_new
        log_likelihood += np.log(pie_1_new * density_1_new + pie_2_new * density_2_new)
    return log_likelihood


if __name__ == '__main__':
    df = pd.read_csv("/Users/geminiwenxu/PycharmProjects/ml_plot_2/data/two_clusters.txt", delimiter=',', header=None,
                     names=['d1', 'd2'])

    plt.scatter(df['d1'], df['d2'])

    mu_1_a = [1, 4]
    mu_2_a = [2, -2]
    K = 2
    sigma_1_a = [[1, 0.5],
                 [0.5, 1]]
    sigma_2_a = [[1, 0],
                 [0, 1]]
    pie_1_a = 0.5
    pie_2_a = 0.5
    var_1_a = multivariate_normal(mean=mu_1_a, cov=sigma_1_a)
    var_2_a = multivariate_normal(mean=mu_2_a, cov=sigma_2_a)

    N = 200
    X = np.linspace(-4, 4, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    pos_1 = np.dstack((X, Y))
    rv_1 = multivariate_normal(mu_1_a, sigma_1_a)
    Z_1 = rv_1.pdf(pos_1)
    pos_2 = np.dstack((X, Y))
    rv_2 = multivariate_normal(mu_2_a, sigma_2_a)
    Z_2 = rv_2.pdf(pos_2)

    # plt.contour(X, Y, Z_1)
    # plt.contour(X, Y, Z_2)
    # plt.show()

    # b
    lst_res_1_b, lst_res_2_b = e_step(df, var_1_a, var_2_a, pie_1_a, pie_2_a)
    plt.scatter(df['d1'], df['d2'], c=lst_res_1_b, cmap='Greens')
    # plt.show()

    # c
    N_1_c = sum(lst_res_1_b)
    N_2_c = sum(lst_res_2_b)
    mu_1_c, mu_2_c, sigma_1_c, sigma_2_c, pie_1_c, pie_2_c = m_step(df, lst_res_1_b, lst_res_2_b, N_1_c,
                                                                    N_2_c)
    print(mu_1_c, mu_2_c, sigma_1_c, sigma_2_c)
    print(sigma_1_c)
    N = 200
    X = np.linspace(-4, 4, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    pos_1 = np.dstack((X, Y))
    rv_1 = multivariate_normal(mu_1_c, sigma_1_c)
    Z_1_c = rv_1.pdf(pos_1)
    pos_2 = np.dstack((X, Y))
    rv_2 = multivariate_normal(mu_2_c, sigma_2_c)
    Z_2_c = rv_2.pdf(pos_2)

    # plt.contour(X, Y, Z_1_c)
    # plt.contour(X, Y, Z_2_c)
    # plt.show()

    # d
    var_1_d = multivariate_normal(mean=mu_1_c, cov=sigma_1_c)
    var_2_d = multivariate_normal(mean=mu_2_c, cov=sigma_2_c)
    lst_res_1_d, lst_res_2_d = e_step(df, var_1_d, var_2_d, pie_1_c, pie_2_c)
    print(lst_res_1_d)
    plt.scatter(df['d1'], df['d2'], c=lst_res_1_d, cmap='Greens')
    # plt.show()

    # e
    N_1_e = sum(lst_res_1_d)
    N_2_e = sum(lst_res_2_d)
    mu_1_e, mu_2_e, sigma_1_e, sigma_2_e, pie_1_e, pie_2_e = m_step(df, lst_res_1_d, lst_res_2_d, N_1_e, N_2_e)
    print(mu_1_e, mu_2_e, sigma_1_e, sigma_2_e)
    N = 200
    X = np.linspace(-4, 4, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    pos_1 = np.dstack((X, Y))
    rv_1 = multivariate_normal(mu_1_e, sigma_1_e)
    Z_1_e = rv_1.pdf(pos_1)
    pos_2 = np.dstack((X, Y))
    rv_2 = multivariate_normal(mu_2_e, sigma_2_e)
    Z_2_e = rv_2.pdf(pos_2)

    # plt.contour(X, Y, Z_1_e)
    # plt.contour(X, Y, Z_2_e)
    # plt.show()

    # f
    var_1_f = multivariate_normal(mean=mu_1_e, cov=sigma_1_e)
    var_2_f = multivariate_normal(mean=mu_2_e, cov=sigma_2_e)
    lst_res_1_f, lst_res_2_f = e_step(df, var_1_f, var_2_f, pie_1_e, pie_2_e)
    plt.scatter(df['d1'], df['d2'], c=lst_res_1_f, cmap='Greens')
    # plt.show()

    # g
    N_1_g = sum(lst_res_1_f)
    N_2_g = sum(lst_res_2_f)
    mu_1_g, mu_2_g, sigma_1_g, sigma_2_g, pie_1_g, pie_2_g = m_step(df, lst_res_1_f, lst_res_2_f, N_1_g, N_2_g)
    print(mu_1_g, mu_2_g, sigma_1_g, sigma_2_g)
    N = 200
    X = np.linspace(-4, 4, N)
    Y = np.linspace(-4, 4, N)
    X, Y = np.meshgrid(X, Y)
    pos_1 = np.dstack((X, Y))
    rv_1 = multivariate_normal(mu_1_g, sigma_1_g)
    Z_1_g = rv_1.pdf(pos_1)
    pos_2 = np.dstack((X, Y))
    rv_2 = multivariate_normal(mu_2_g, sigma_2_g)
    Z_2_g = rv_2.pdf(pos_2)

    plt.contour(X, Y, Z_1_g)
    plt.contour(X, Y, Z_2_g)
    plt.show()

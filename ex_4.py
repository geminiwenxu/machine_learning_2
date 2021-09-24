import numpy as np
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


def e_step(df, mu_1, mu_2, sigma_1, sigma_2, pie_1, pie_2):
    var_1 = multivariate_normal(mean=mu_1, cov=sigma_1)
    var_2 = multivariate_normal(mean=mu_2, cov=sigma_2)
    lst_res_1 = []
    lst_res_2 = []
    L_old = 0
    for index, row in df.iterrows():
        density_1 = var_1.pdf([row['d1'], row['d2']])
        density_2 = var_2.pdf([row['d1'], row['d2']])
        p_x = pie_1 * density_1 + pie_2 * density_2
        res_1 = np.float32((pie_1 * density_1) / p_x).item()
        lst_res_1.append(res_1)
        res_2 = np.float32((pie_2 * density_2) / p_x).item()
        lst_res_2.append(res_2)
        log_pz_1 = np.log(pie_1 + 1e-20)
        log_pz_2 = np.log(pie_2 + 1e-20)
        L_old += np.multiply(res_1, log_pz_1) + np.multiply(res_1,
                                                            np.log(density_1) + np.multiply(res_1, -np.log(res_1)) +
                                                            np.multiply(res_2, log_pz_2) + np.multiply(res_2, np.log(
                                                                density_2)) + np.multiply(res_2, -np.log(res_2)))
    print("L at old parameters", L_old)
    return lst_res_1, lst_res_2, L_old


def m_step(df, lst_res_1, lst_res_2):
    true_log_likelihood = 0
    N_1 = sum(lst_res_1)
    N_2 = sum(lst_res_2)
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

    var_1_new = multivariate_normal(mean=mu_1_new, cov=sigma_1_new)
    var_2_new = multivariate_normal(mean=mu_2_new, cov=sigma_2_new)
    L_new = 0
    for index, row in df.iterrows():
        density_1_new = var_1_new.pdf([row['d1'], row['d2']])
        density_2_new = var_2_new.pdf([row['d1'], row['d2']])
        true_log_likelihood += np.log(pie_1_new * density_1_new + pie_2_new * density_2_new)
        log_pz_1 = np.log(pie_1 + 1e-20)
        log_pz_2 = np.log(pie_2 + 1e-20)
        L_new += np.multiply(lst_res_1[index], log_pz_1) + np.multiply(lst_res_1[index],
                                                                       np.log(density_1_new) + np.multiply(
                                                                           lst_res_1[index],
                                                                           -np.log(lst_res_1[index])) +
                                                                       np.multiply(lst_res_2[index],
                                                                                   log_pz_2) + np.multiply(
                                                                           lst_res_2[index], np.log(
                                                                               density_2_new)) + np.multiply(
                                                                           lst_res_2[index], -np.log(lst_res_2[index])))
    print("L at new parameters", L_new)
    print("true log likelihood", true_log_likelihood)
    return mu_1_new, mu_2_new, sigma_1_new, sigma_2_new, pie_1_new, pie_2_new, L_new, true_log_likelihood


if __name__ == '__main__':
    df = pd.read_csv("/data/two_clusters.txt", delimiter=',', header=None,
                     names=['d1', 'd2'])
    mu_1 = [1, 1]
    mu_2 = [2, 2]
    sigma_1 = [[1, 0.5], [0.5, 1]]
    sigma_2 = [[1, 0], [0, 1]]
    pie_1 = 0.5
    pie_2 = 0.5
    lst_L_old = []
    lst_L_new = []
    lst_true_log_likelihood = []
    steps = 8
    for i in range(steps):
        lst_res_1, lst_res_2, L_old = e_step(df, mu_1, mu_2, sigma_1, sigma_2, pie_1, pie_2)
        mu_1_new, mu_2_new, sigma_1_new, sigma_2_new, pie_1_new, pie_2_new, L_new, true_log_likelihood = m_step(df,
                                                                                                                lst_res_1,
                                                                                                                lst_res_2)
        lst_L_old.append(L_old)
        lst_L_new.append(L_new)
        lst_true_log_likelihood.append(true_log_likelihood)
        mu_1 = mu_1_new
        mu_2 = mu_2_new
        sigma_1 = sigma_1_new
        sigma_2 = sigma_2_new
        pie_1 = pie_1_new
        pie_2 = pie_2_new

    plt.plot(np.linspace(1, steps, steps, endpoint=True), lst_L_old, 'blue', label='L at E step')
    plt.plot(np.linspace(1, steps, steps, endpoint=True), lst_L_new, 'red', label='L at M step')
    plt.plot(np.linspace(1, steps, steps, endpoint=True), lst_true_log_likelihood, 'black', label='true log likelihood')
    plt.xlabel("steps")
    plt.ylabel("value of L and log-likelihood")
    plt.legend()
    plt.show()

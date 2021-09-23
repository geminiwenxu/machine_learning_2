import numpy as np
# from sklearn.cluster import KMeans
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


def first_step(df, mu_1, mu_2):
    for index, row in df.iterrows():
        row_index = df.index[index]

        dist_1 = np.float32((row['d1'] - mu_1['d1']) ** 2 + (row['d2'] - mu_1['d2']) ** 2).item()
        dist_2 = np.float32((row['d1'] - mu_2['d1']) ** 2 + (row['d2'] - mu_2['d2']) ** 2).item()
        df.loc[row_index, 'dist_1'] = dist_1
        df.loc[row_index, 'dist_2'] = dist_2
        if dist_1 < dist_2:
            df.loc[row_index, 'cluster'] = 1
        else:
            df.loc[row_index, 'cluster'] = 2

    df1 = df[df.cluster == 1]
    df2 = df[df.cluster == 2]
    plt.scatter(df1.d1, df1.d2, color="green")
    plt.scatter(df2.d1, df2.d2, color="red")
    plt.plot(mu_1['d1'], mu_1['d2'], color='darkgreen', marker='X')
    plt.plot(mu_2['d1'], mu_2['d2'], color='darkred', marker='x')
    plt.show()
    return df1, df2


def second_step(df1, df2):
    mu_1_1 = (np.float32(df1['d1'].sum()).item()) / len(df1.index)+0.0000001
    mu_1_2 = np.float32(df1['d2'].sum()).item() / len(df1.index)+0.0000001
    mu_1 = pd.DataFrame([[mu_1_1, mu_1_2]], columns=['d1', 'd2'])

    mu_2_1 = (np.float32(df2['d1'].sum()).item()) / len(df2.index)+0.0000001
    mu_2_2 = np.float32(df2['d2'].sum()).item() / len(df2.index)+0.0000001
    mu_2 = pd.DataFrame([[mu_2_1, mu_2_2]], columns=['d1', 'd2'])
    plt.plot(mu_1['d1'], mu_1['d2'], color='darkgreen', marker='X')
    plt.plot(mu_2['d1'], mu_2['d2'], color='darkred', marker='X')
    return mu_1, mu_2


if __name__ == '__main__':
    df = pd.read_csv("/data/two_clusters.txt", delimiter=',', header=None,
                     names=['d1', 'd2'])

    plt.scatter(df['d1'], df['d2'])
    plt.plot(1.2610673e+00,   8.7850842e-02, color='darkgreen', marker='X')
    plt.plot(1.4257554e+00,  -2.1140316e-01, color='darkred', marker='X')
    plt.show()

    df1, df2 = first_step(df, mu_1=df.iloc[0], mu_2=df.iloc[5])
    mu_1, mu_2 = second_step(df1, df2)

    new_df1, new_df2 = first_step(df, mu_1, mu_2)
    new_mu_1, new_mu_2 = second_step(new_df1, new_df2)

    first_step(df,  new_mu_1, new_mu_2)

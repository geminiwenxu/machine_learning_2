import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Dense, Input
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras.backend as K
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from tensorflow.keras import initializers

if __name__ == '__main__':
    # a
    df = pd.read_csv("/data/data_ex5", delimiter=',', header=None,
                     names=['d1', 'd2', 'd3'])

    X_std = StandardScaler().fit_transform(df)

    # c
    input_layer = Input(shape=(3,))
    code = Dense(1, activation=None, use_bias=True)(input_layer)
    output_layer = Dense(3, activation=None, use_bias=True)(code)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_std, X_std, batch_size=150, epochs=500)
    plt.plot(history.history['loss'], label='train loss')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()
    print(K.eval(autoencoder.optimizer.lr))

    # d
    """alternative way to get hidden value out"""
    model1 = Model(input_layer, code)
    print(autoencoder.get_weights())
    print(autoencoder.layers[1].get_weights())
    model1.set_weights(autoencoder.layers[1].get_weights())
    model1.compile(optimizer='sgd', loss='mse')
    hidden_code = model1.predict(X_std)
    # print(hidden_code)
    weights = autoencoder.get_weights()
    W_e = weights[0]
    print("W_e", W_e)

    """using bankend function to get hidden value out"""
    # get_hidden_layer_output = K.function([autoencoder.layers[0].input], [autoencoder.layers[1].output])
    # layer_output = get_hidden_layer_output([X_std])
    # print(len(layer_output[0]))

    cov_mat = np.cov(X_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print("eigen value", eig_vals)
    print("PC eigen vector", eig_vecs)
    projection = np.dot(X_std, eig_vecs[0])
    print(len(projection))
    for i in range(150):
        plt.scatter(hidden_code[i], projection[i])
    plt.xlabel('hidden')
    plt.ylabel('x projection on pc vector')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()

    # e
    # X_train, X_test = train_test_split(X_std, test_size=0.2, random_state=42)
    # X_train_2, X_test_2 = train_test_split(X_std, test_size=0.2, random_state=0, shuffle=True)
    # input_layer = Input(shape=(3,))
    # code = Dense(2, activation=None, use_bias=True)(input_layer)
    # output_layer = Dense(3, activation=None)(code)
    #
    # autoencoder_2 = Model(input_layer, output_layer)
    # autoencoder_2.compile(optimizer='sgd', loss='mse')
    # autoencoder_2.fit(X_train, X_train, batch_size=150, epochs=300)
    # """alternative way to get hidden unit output"""
    # model2 = Model(input_layer, code)
    #
    # model2.set_weights(autoencoder_2.layers[1].get_weights())
    # model2.compile(optimizer='sgd', loss='mse')
    # hidden_code_1 = model2.predict(X_train)
    # hidden_code_2 = model2.predict(X_train_2)
    # for i in range(120):
    #     unit_1 = hidden_code_1[i][0]
    #     unit_2 = hidden_code_1[i][1]
    #     plt.scatter(unit_1, unit_2, color='red', marker='X')
    #     unit_1_1 = hidden_code_2[i][0]
    #     unit_2_2 = hidden_code_2[i][1]
    #     plt.scatter(unit_1_1, unit_2_2, color='blue')
    # plt.xlabel('unit 1')
    # plt.ylabel('unit 2')
    # plt.show()
    """using backend to get hidden unit out"""
    # get_hidden_layer_output_2 = K.function([autoencoder_2.layers[0].input], [autoencoder_2.layers[1].output])
    # layer_output_2 = get_hidden_layer_output_2([X_std])
    #
    # autoencoder_2.fit(X_train_2, X_train_2, batch_size=150, epochs=300)
    # get_hidden_layer_output_2_2 = K.function([autoencoder_2.layers[0].input], [autoencoder_2.layers[1].output])
    # layer_output_2_2 = get_hidden_layer_output_2([X_std])
    # for i in range(120):
    #     unit_1 = layer_output_2[0][i][0]
    #     unit_2 = layer_output_2[0][i][1]
    #     plt.scatter(unit_1, unit_2, color='red')
    #     unit_1_2 = layer_output_2_2[0][i][0]
    #     unit_2_2 = layer_output_2_2[0][i][1]
    #     plt.scatter(unit_1_2, unit_2_2, color='blue')
    # plt.xlabel('unit 1')
    # plt.ylabel('unit 2')
    # plt.show()

    # f
    # linear_error = []
    # for i in range(5):
    #     X_train, X_test = train_test_split(X_std, test_size=0.2, random_state=42)
    #     input_layer = Input(shape=(3,))
    #     code = Dense(2, activation=None, use_bias=True)(input_layer)
    #     output_layer = Dense(3, activation=None)(code)
    #     autoencoder = Model(input_layer, output_layer)
    #     autoencoder.compile(optimizer='sgd', loss='mse')
    #     history = autoencoder.fit(X_train, X_train, batch_size=150, epochs=500)
    #     linear_error.append(autoencoder.evaluate(X_test, X_test))
    #     plt.plot(history.history['loss'], label='train loss')
    #     plt.xlabel("iterations")
    #     plt.ylabel("loss")
    # plt.show()


    # error = []
    # for i in range(5):
    #     X_train, X_test = train_test_split(X_std, test_size=0.2, random_state=42)
    #     input_layer = Input(shape=(3,))
    #     code = Dense(2, activation='sigmoid', use_bias=True)(input_layer)
    #     output_layer = Dense(3, activation='sigmoid')(code)
    #     autoencoder = Model(input_layer, output_layer)
    #     autoencoder.compile(optimizer='adam', loss='mse')
    #     history = autoencoder.fit(X_train, X_train, batch_size=150, epochs=10000)
    #     error.append(autoencoder.evaluate(X_test, X_test))
    #     plt.plot(history.history['loss'], label='train loss')
    #     plt.xlabel("iterations")
    #     plt.ylabel("loss")
    # plt.show()
    # print(linear_error, sum(linear_error)/5)
    # print(error, sum(error)/5)

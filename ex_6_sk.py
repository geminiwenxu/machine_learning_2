from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt


def sample():
    x = np.random.uniform(low=-1.0, high=1.0, size=50)
    # t = np.sin(x)
    t = x ** 2
    return x, t


if __name__ == "__main__":
    x_train, t_train = sample()
    plt.scatter(x_train, t_train, color='pink')
    ls_loss = []

    loss = np.inf
    for i, x in enumerate(x_train):
        while loss > 0.1:
            t = x ** 2
            regr = MLPRegressor(hidden_layer_sizes=(3,), activation="tanh", solver="sgd", alpha=0.01).fit(
                x.reshape(1, -1), t.reshape(1, -1))
            loss = regr.loss_
            ls_loss.append(loss)
    print(ls_loss)
    print(len(ls_loss))

    # plt.plot(np.linspace(-1, 1, 100), regr.predict(np.linspace(-1, 1, 100).reshape(-1, 1)))
    #
    # if loss <= 0.5:
    #     # plt.plot(np.linspace(1, len(loss), len(loss)), loss)
    #     # plt.xlabel("iteration")
    #     # plt.ylabel("loss")
    #     weights = regr.coefs_
    #     h = np.tanh(np.multiply(x_train.reshape(-1, 1), weights[0]))
    #     h1 = []
    #     h2 = []
    #     h3 = []
    #     for i in range(50):
    #         h1.append(h[i][0])
    #         h2.append(h[i][1])
    #         h3.append(h[i][2])
    #     plt.scatter(x_train, h1, color="red")
    #     plt.scatter(x_train, h2, color="blue")
    #     plt.scatter(x_train, h3, color='green')
    #     # plt.title('loss')
    #     plt.show()

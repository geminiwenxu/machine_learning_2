import numpy as np
import matplotlib.pyplot as plt


def sample():
    x = np.random.uniform(low=-1.0, high=1.0, size=20000)
    t = np.sin(x)
    # t = x ** 2
    return x, t


def forward(w1, w2, x):
    a1 = np.dot(w1, x)
    z1 = np.tanh(a1)
    y = np.dot(z1.T, w2)
    return a1, z1, y


def predict(w1, w2, x):
    return forward(w1, w2, x.T)[2].T


def backward(w2, z1, y, t):
    delta2 = np.absolute(y - t)
    delta1 = ((1 - z1) ** 2) * np.dot(w2, delta2)
    return delta1, delta2


def loss(y, t):
    return 0.5 * (y - t) ** 2


def train_step(alpha, w1, w2, x, t):
    a1, z1, y = forward(w1, w2, x, )
    delta1, delta2 = backward(w2, z1, y, t.T)
    gradw1 = np.dot(delta1, x)
    gradw2 = np.dot(z1, delta2)
    return w1 - alpha * gradw1, w2 - alpha * gradw2


if __name__ == "__main__":
    x_train, t_train = sample()
    plt.scatter(x_train, t_train, color='red')
    err_train = []
    w1 = np.random.rand(3, 1) * 1e-3
    w2 = np.random.rand(3, 1) * 1e-3
    # w1 = np.full((3, 1), 0.001)
    # w2 = np.full((3, 1), 0.001)
    alpha = 0.001
    current_loss = 0
    acc_loss = np.inf
    for j, x in enumerate(x_train):
        while acc_loss > 0.1:
            acc_loss = float(loss(predict(w1, w2, x), t_train[j])[0][0])
            # err_train.append(loss(predict(w1, w2, x), t_train[j]))
            err_train.append(acc_loss)
            w1, w2 = train_step(alpha, w1, w2, x, t_train[j])
    print("final w1 and w2", w1, w2)
    print("loss", err_train)
    # plt.plot(np.arange(len(err_train)), err_train)
    y_pred = []
    for i, x in enumerate(x_train):
        y = predict(w1, w2, x)
        y_pred.append(y)
    plt.plot(x_train, np.array(y_pred).ravel(), color="pink")

    h1 = np.tanh(np.multiply(x_train.T, w1[0]))
    h2 = np.tanh(np.multiply(x_train.T, w1[1]))
    h3 = np.tanh(np.multiply(x_train.T, w1[2]))
    # plt.plot(x_train, h1, color="blue")
    # plt.plot(x_train, h2, color="red")
    # plt.plot(x_train, h3, color="green")
    plt.show()

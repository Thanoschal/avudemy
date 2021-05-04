import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def start():
    np.random.seed(0)
    points = 500
    X = np.linspace(-3, 3, points)
    Y = np.sin(X) + np.random.uniform(-0.5, 0.5, points)

    model = Sequential()
    model.add(Dense(50, input_dim=1, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(1))

    adam = Adam(lr=0.01)
    model.compile(loss='mse', optimizer=adam)
    model.fit(X, Y , epochs=50)

    predictions = model.predict(X)
    plt.scatter(X, Y)
    plt.plot(X, predictions, 'ro')

    plt.show()

if __name__ == '__main__':
    start()
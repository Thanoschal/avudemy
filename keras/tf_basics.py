import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def start():
    n_pts = 500
    np.random.seed(0)
    Xa = np.array([np.random.normal(13, 2, n_pts),
                   np.random.normal(12, 2, n_pts)]).T
    Xb = np.array([np.random.normal(8, 2, n_pts),
                   np.random.normal(6, 2, n_pts)]).T

    X = np.vstack((Xa, Xb))
    Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T

    model = Sequential()
    #units: output nodes
    #input_shape: input nodes
    model.add(Dense(units = 1, input_shape = (2,), activation='sigmoid'))
    adam = Adam(lr = 0.1)
    model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
    #batch_size = number of data points every epoch
    #epochs = how many times to iterate the dataset
    h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs = 100, shuffle = 'true')

    plt.plot(h.history['accuracy'])
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.legend(['accuracy'])
    plt.show()
    plt.clf()

    plt.plot(h.history['loss'])
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend(['loss'])
    plt.show()

    point = np.array([[7.5, 5]])
    prediction = model.predict(point)
    print(prediction)


if __name__ == '__main__':
    start()
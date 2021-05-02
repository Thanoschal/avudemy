import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn import datasets


def start():
    n_pts = 500
    #ready dataset from sklearn, factor is the diameter ratio between circles
    X, y = datasets.make_circles(n_samples=n_pts, random_state= 123, noise=0.1, factor = 0.2)
    plt.scatter(X[y == 0, 0], X[y == 0, 1])
    plt.scatter(X[y == 1, 0], X[y == 1, 1])
    plt.show()
    plt.clf()
    model = Sequential()
    #input_shape: input nodes
    model.add(Dense(4, input_shape = (2,), activation='sigmoid'))
    #add output layer
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr = 0.3)
    model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
    #batch_size = number of data points every epoch
    #epochs = how many times to iterate the dataset
    h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs = 10, shuffle = 'true')


    plt.plot(h.history['accuracy'])
    plt.title('accuracy')
    plt.xlabel('epochs')
    plt.legend(['accuracy'])
    plt.show()

    """
    plt.plot(h.history['loss'])
    plt.title('loss')
    plt.xlabel('epochs')
    plt.legend(['loss'])
    plt.show()

    point = np.array([[7.5, 5]])
    prediction = model.predict(point)
    print(prediction)
    """




if __name__ == '__main__':
    start()
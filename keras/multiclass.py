import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn import datasets
from keras.utils.np_utils import to_categorical


def start():
    n_pts = 500
    centers = [[-1,1], [-1,-1], [1,-1]]
    #ready dataset from sklearn, factor is the diameter ratio between circles
    X, y = datasets.make_blobs(n_samples=n_pts, random_state= 123, centers=centers, cluster_std=0.4)

    y_cat = to_categorical(y, 3)
    print(y_cat)


    model = Sequential()
    #input_shape: input nodes
    model.add(Dense(units=3, input_shape = (2,), activation='softmax'))
    #add output layer
    adam = Adam(lr = 0.3)
    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #batch_size = number of data points every epoch
    #epochs = how many times to iterate the dataset
    h = model.fit(x=X, y=y_cat, verbose=1, batch_size=50, epochs = 30, shuffle = 'true')


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
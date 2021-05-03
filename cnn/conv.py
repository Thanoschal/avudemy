import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import  Conv2D
from keras.layers.convolutional import  MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random


def start():
    np.random.seed(0)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)
    print(X_test.shape)
    assert (X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
    assert (X_train.shape[1:] == (28, 28)), "The dimensions of the images are not 28 x 28."
    assert (X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
    assert (X_test.shape[1:] == (28, 28)), "The dimensions of the images are not 28 x 28."

    num_of_samples = []

    cols = 5
    num_classes = 10

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 10))
    fig.tight_layout()

    for i in range(cols):
        for j in range(num_classes):
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j))
                num_of_samples.append(len(x_selected))

    """
    print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_samples)
    plt.title("Distribution of the train dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    """


    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    X_train = X_train / 255
    X_test = X_test / 255

    #define the LeNet model function
    def leNet_model():
        model = Sequential()
        #number of filters: 30
        # 5x5 filter
        # strides : how much the kernel moves for each convolution
        # padding: valid,  casual, same
        #input shape is just for the first layer
        #PADDING NEEDED WHEN FREATURES ARE IN THE CORNER OF THE IMAGES
        model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    model = leNet_model()
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)



if __name__ == '__main__':
    start()
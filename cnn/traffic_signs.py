import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
np.random.seed(0)


def start():


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
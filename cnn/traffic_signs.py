import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import random
import cv2
np.random.seed(0)


def start():

    with open('../german-traffic-signs/train.p', 'rb') as f:
        train_dat = pickle.load(f)
    with open('../german-traffic-signs/valid.p', 'rb') as f:
        val_dat = pickle.load(f)
    with open('../german-traffic-signs/test.p', 'rb') as f:
        test_dat = pickle.load(f)

    X_train , y_train = train_dat['features'], train_dat['labels']
    X_val, y_val = val_dat['features'], val_dat['labels']
    X_test, y_test = test_dat['features'], test_dat['labels']

    data = pd.read_csv('../german-traffic-signs/signnames.csv')

    #grayscale the data, because in our case color is not supplying
    #concrete information about the classification
    def grayscale(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    #equalize the lighting and contrast of our images
    def equalize(img):
        cv2.equalizeHist(img)
        return img

    def preprocessing(img):
        img = grayscale(img)
        img = equalize(img)
        img = img / 255
        return img

    #iterate through all images in X_train
    #apply the preprocessing function
    #convert the output of map to a list and then in numpy array
    #store it back in X_train
    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))

    #plt.imshow(X_train[random.randint(0, len(X_train - 1))], cmap='gray')
    #plt.show()

    #add a depth of 1 because the conv layer needs it
    X_train = X_train.reshape(34799, 32, 32, 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    #one-hot-encode the labels of the sets
    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)
    y_test = to_categorical(y_test, 43)

    datagen = ImageDataGenerator(width_shift_range=0.1,
                       height_shift_range=0.1,
                       zoom_range=0.2,
                       shear_range=0.1,
                       rotation_range=10)

    datagen.fit(X_train)
    batches = datagen.flow(X_train, y_train, batch_size=15)
    X_batch, y_batch = next(batches)


    """
    num_of_samples = []

    cols = 5
    num_classes = 43

    fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
    fig.tight_layout()
    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train == j]
            axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
            axs[j][i].axis("off")
            if i == 2:
                axs[j][i].set_title(str(j) + "-" + row['SignName'])
                num_of_samples.append(len(x_selected))
    plt.show()
    """

    #define the LeNet model function
    def leNet_model():
        model = Sequential()
        #number of filters: 30
        # 5x5 filter
        # strides : how much the kernel moves for each convolution
        # padding: valid,  casual, same
        #input shape is just for the first layer
        #PADDING NEEDED WHEN FREATURES ARE IN THE CORNER OF THE IMAGES
        model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
        model.add(Conv2D(60, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(Conv2D(30, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(43, activation='softmax'))
        model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    model = leNet_model()
    print(model.summary())

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50),
                                  steps_per_epoch=2000,
                                  epochs=10,
                                  validation_data=(X_val, y_val),
                                  shuffle=1)

    score = model.evaluate(X_test, y_test, verbose= 0)
    print('Test score: ', score[0])
    print('Test accuracy: ', score[1])


if __name__ == '__main__':
    start()
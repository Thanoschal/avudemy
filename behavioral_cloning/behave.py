import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import pandas as pd
import ntpath
import matplotlib.pyplot as plt

def start():
    #name the columns of the data frame
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv('../bc_data/driving_log.csv', names=columns)
    #keep all the path
    pd.set_option('display.max_colwidth',None)

    #keep only the last part of the path
    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)

    #show the histogram to analyze the data
    num_bins = 25
    thresh_samples_per_bin = 200
    hist, bins = np.histogram(data['steering'], num_bins)
    print(bins)
    #center the values around 0
    center = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(center , hist, width=0.05)
    plt.show()


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


if __name__ == '__main__':
    start()
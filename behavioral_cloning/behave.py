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
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv('../bc_data/driving_log.csv', names=columns)
    pd.set_option('display.max_colwidth',None)

    data['center'] = data['center'].apply(path_leaf)
    data['left'] = data['left'].apply(path_leaf)
    data['right'] = data['right'].apply(path_leaf)

    num_bins = 25
    hist, bins = np.histogram(data['steering'], num_bins)
    print(bins)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail


if __name__ == '__main__':
    start()
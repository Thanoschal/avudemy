import numpy as np


def start():

    #creates an array
    x = np.arange(4).reshape(2, 2)
    y = np.arange(4, 8).reshape(2, 2)

    z = np.hstack((x,y)) #horizontal stack
    w = np.vstack((x, y))  # vertical stack

    d = np.concatenate((x, y), axis = 0) #vertical stack 0: rows, 1: columns




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
import numpy as np


def start():

    #creates an array
    y = np.arange(18).reshape(3, 2, 3)

    print(y[1, 0:2, 0:3])
    print(y[y>5])

    #x.max()
    #x.min()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
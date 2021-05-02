import numpy as np


def start():

    #creates an array
    x = np.arange(9).reshape(-1,3) #-1 denotes an unknown dimension
    y = np.arange(18).reshape(2, 3, 3)  # -1 denotes an unknown dimension
    print(y)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
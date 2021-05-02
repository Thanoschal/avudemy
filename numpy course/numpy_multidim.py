import numpy as np


def start():

    #creates an array
    x = np.arange(0,3) #row1
    y = np.arange(3,6) #row2
    z = np.arange(6,9) #row3

    multi_array = np.array([x,y,z], dtype = np.uint16)

    print(multi_array)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
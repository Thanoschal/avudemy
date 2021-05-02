import numpy as np


def start():

    #creates an array
    x = np.arange(1,10)

    print(x[2:7]) #the stop index is exclusive
    print(x[2:7:2])  # step size 2
    print(x[:7])  # from the start to 7
    print(x[2:])  # from 2 to the end

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
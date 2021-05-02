import numpy as np


def start():

    #creates an array
    mat_a = np.matrix([0,3,5,5,5,2]).reshape(2,3)
    mat_b = np.matrix([3,4,3,-2,4,-2]).reshape(3, 2)

    product = np.matmul(mat_a, mat_b)

    print(product)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
import numpy as np


def start():

    #creates an array
    x = np.arange(18).reshape(3, 2, 3)

    rav_x = x.ravel() #a view of the array
    print(rav_x)    #if i change a value in the raveled array
                    #it will also change in the original array

    flat_x = x.flatten() #copy and allocates memory
    print(flat_x)

    #change the shape directly
    y = np.arange(9)
    y.shape = [3,3]
    print(y)

    #transpose
    print(y.transpose())
    print(y.T)

    #resize
    print(np.resize(y, (6,6)))
    #data is repeated if there is not enough of it

    print(np.zeros((2,3), dtype =int))
    print(np.ones((2, 3), dtype=int))
    print(np.eye(2, 3))

    print(np.random.rand(4, 4))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
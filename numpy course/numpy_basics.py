import numpy as np
import time

def start():

    list_two = list(range(1,100000))
    list_three = list(range(1, 100000))
    list_sum = []

    start_time_for = time.time()

    for index in range(3):
        list_two[index] =  list_two[index] ** 2
        list_three[index] =  list_three[index] ** 3
        list_sum.append(list_two[index] + list_three[index])

    print("--- %.10f seconds ---" % (time.time() - start_time_for))

    start_time_numpy = time.time()

    #creates an array
    array_two = np.arange(1,100000) ** 2
    array_three = np.arange(1,100000) ** 3
    sum_array = array_two + array_three

    print("--- %.10f seconds ---" % (time.time() - start_time_numpy))

    np.power(np.array([1,2,3]),4)
    print(np.negative(np.array([1,2,3])))

    print(np.exp(np.array([1, 2, 3])))

    print(np.log(np.array([1, 2, 3])))

    print(np.sin(np.array([1, 2, 3])))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
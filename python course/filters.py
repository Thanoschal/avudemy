# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def ever_or_odd(num):
    return num % 2 == 0

def start():
    numbers = [1,2,3,4,5]
    filtered_list = list(filter(ever_or_odd,numbers))
    print(filtered_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

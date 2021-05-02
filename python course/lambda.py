
def start():
    numbers = [1,2,3,4,5]
    filtered_list = list(filter(lambda num: num % 2 == 0, numbers))
    print(filtered_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
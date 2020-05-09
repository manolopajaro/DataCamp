def test1():
    arr = [1]
    arr_minus = arr[:-1]
    arr_plus = arr[1:]
    print(arr)


def test2(input_list):
    return max([element * adjacent for element, adjacent in zip(input_list[:-1], input_list[1:0])])


if __name__ == "__main__":
    test2([])

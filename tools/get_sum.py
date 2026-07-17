from langchain.tools import tool


@tool
def get_sum(a: int, b: int) -> int:
    """
    get the summation of two integers
    :param a: int, input integer
    :param b: int, input integer
    :return: int, the sum of a and b
    """
    return a + b


if __name__ == "__main__":
    result = get_sum.invoke({"a": 2, "b": 3})
    print(result)

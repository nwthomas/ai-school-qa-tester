
def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    assert factorial(10) == 362880
    try:
        factorial(-1)
    except ValueError:
        pass
    else:
        assert False, "ValueError not raised for negative input"

if __name__ == '__main__':
    test_factorial()
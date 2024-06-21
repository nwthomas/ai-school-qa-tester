import pytest
from factorial import factorial

def test_factorial_wrong_data_type():
    with pytest.raises(TypeError):
        factorial('a')
    with pytest.raises(TypeError):
        factorial(3.5)
    with pytest.raises(TypeError):
        factorial([])
    with pytest.raises(TypeError):
        factorial({})

if __name__ == '__main__':
    test_factorial_wrong_data_type()

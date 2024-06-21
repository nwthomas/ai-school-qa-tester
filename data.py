training_data = [
    {
        "prompt": "```python\ndef add(a, b):\n    return a + b\n```",
        "completion": "```python\nimport unittest\n\nclass TestAddFunction(unittest.TestCase):\n    def test_add_large_numbers(self):\n        self.assertEqual(add(1e10, 1e10), 2e10)\n\n    def test_add_float_numbers(self):\n        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=7)\n\n    def test_add_string_numbers(self):\n        with self.assertRaises(TypeError):\n            add('1', 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef multiply(a, b):\n    return a * b\n```",
        "completion": "```python\nimport unittest\n\nclass TestMultiplyFunction(unittest.TestCase):\n    def test_multiply_large_numbers(self):\n        self.assertEqual(multiply(1e10, 1e10), 1e20)\n\n    def test_multiply_by_float_numbers(self):\n        self.assertAlmostEqual(multiply(0.1, 0.2), 0.02, places=7)\n\n    def test_multiply_by_string_numbers(self):\n        with self.assertRaises(TypeError):\n            multiply('1', 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n```",
        "completion": "```python\nimport unittest\n\nclass TestDivideFunction(unittest.TestCase):\n    def test_divide_large_numbers(self):\n        self.assertAlmostEqual(divide(1e10, 1e2), 1e8)\n\n    def test_divide_by_float_numbers(self):\n        self.assertAlmostEqual(divide(0.1, 0.2), 0.5, places=7)\n\n    def test_divide_by_zero_float(self):\n        with self.assertRaises(ValueError):\n            divide(1.0, 0.0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef subtract(a, b):\n    return a - b\n```",
        "completion": "```python\nimport unittest\n\nclass TestSubtractFunction(unittest.TestCase):\n    def test_subtract_large_numbers(self):\n        self.assertEqual(subtract(1e10, 1e9), 9e9)\n\n    def test_subtract_float_numbers(self):\n        self.assertAlmostEqual(subtract(0.1, 0.2), -0.1, places=7)\n\n    def test_subtract_string_numbers(self):\n        with self.assertRaises(TypeError):\n            subtract('1', 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef power(a, b):\n    return a ** b\n```",
        "completion": "```python\nimport unittest\n\nclass TestPowerFunction(unittest.TestCase):\n    def test_power_large_numbers(self):\n        self.assertEqual(power(10, 10), 1e10)\n\n    def test_power_zero_exponent(self):\n        self.assertEqual(power(2, 0), 1)\n\n    def test_power_negative_exponent(self):\n        self.assertAlmostEqual(power(2, -1), 0.5, places=7)\n\n    def test_power_float_exponent(self):\n        self.assertAlmostEqual(power(9, 0.5), 3, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef modulo(a, b):\n    return a % b\n```",
        "completion": "```python\nimport unittest\n\nclass TestModuloFunction(unittest.TestCase):\n    def test_modulo_positive_numbers(self):\n        self.assertEqual(modulo(10, 3), 1)\n\n    def test_modulo_negative_numbers(self):\n        self.assertEqual(modulo(-10, 3), -1)\n\n    def test_modulo_by_zero(self):\n        with self.assertRaises(ZeroDivisionError):\n            modulo(10, 0)\n\n    def test_modulo_float_numbers(self):\n        self.assertAlmostEqual(modulo(10.5, 3.2), 0.9, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFactorialFunction(unittest.TestCase):\n    def test_factorial_zero(self):\n        self.assertEqual(factorial(0), 1)\n\n    def test_factorial_positive_number(self):\n        self.assertEqual(factorial(5), 120)\n\n    def test_factorial_large_number(self):\n        self.assertEqual(factorial(10), 3628800)\n\n    def test_factorial_negative_number(self):\n        with self.assertRaises(RecursionError):\n            factorial(-1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef fibonacci(n):\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFibonacciFunction(unittest.TestCase):\n    def test_fibonacci_zero(self):\n        self.assertEqual(fibonacci(0), 0)\n\n    def test_fibonacci_one(self):\n        self.assertEqual(fibonacci(1), 1)\n\n    def test_fibonacci_positive_number(self):\n        self.assertEqual(fibonacci(5), 5)\n\n    def test_fibonacci_large_number(self):\n        self.assertEqual(fibonacci(10), 55)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsPrimeFunction(unittest.TestCase):\n    def test_is_prime_small_prime(self):\n        self.assertTrue(is_prime(7))\n\n    def test_is_prime_small_non_prime(self):\n        self.assertFalse(is_prime(4))\n\n    def test_is_prime_large_prime(self):\n        self.assertTrue(is_prime(29))\n\n    def test_is_prime_large_non_prime(self):\n        self.assertFalse(is_prime(30))\n\n    def test_is_prime_negative_number(self):\n        self.assertFalse(is_prime(-5))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsEvenFunction(unittest.TestCase):\n    def test_is_even_positive_number(self):\n        self.assertTrue(is_even(4))\n\n    def test_is_even_negative_number(self):\n        self.assertTrue(is_even(-4))\n\n    def test_is_even_odd_number(self):\n        self.assertFalse(is_even(5))\n\n    def test_is_even_zero(self):\n        self.assertTrue(is_even(0))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsPalindromeFunction(unittest.TestCase):\n    def test_is_palindrome_even_length(self):\n        self.assertTrue(is_palindrome('abba'))\n\n    def test_is_palindrome_odd_length(self):\n        self.assertTrue(is_palindrome('racecar'))\n\n    def test_is_palindrome_not_palindrome(self):\n        self.assertFalse(is_palindrome('hello'))\n\n    def test_is_palindrome_empty_string(self):\n        self.assertTrue(is_palindrome(''))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        "completion": "```python\nimport unittest\n\nclass TestReverseStringFunction(unittest.TestCase):\n    def test_reverse_string_regular(self):\n        self.assertEqual(reverse_string('hello'), 'olleh')\n\n    def test_reverse_string_empty(self):\n        self.assertEqual(reverse_string(''), '')\n\n    def test_reverse_string_single_character(self):\n        self.assertEqual(reverse_string('a'), 'a')\n\n    def test_reverse_string_palindrome(self):\n        self.assertEqual(reverse_string('racecar'), 'racecar')\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef find_max(arr):\n    return max(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFindMaxFunction(unittest.TestCase):\n    def test_find_max_positive_numbers(self):\n        self.assertEqual(find_max([1, 2, 3, 4, 5]), 5)\n\n    def test_find_max_negative_numbers(self):\n        self.assertEqual(find_max([-1, -2, -3, -4, -5]), -1)\n\n    def test_find_max_mixed_numbers(self):\n        self.assertEqual(find_max([1, -2, 3, -4, 5]), 5)\n\n    def test_find_max_single_element(self):\n        self.assertEqual(find_max([1]), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef sum_list(lst):\n    return sum(lst)\n```",
        "completion": "```python\nimport unittest\n\nclass TestSumListFunction(unittest.TestCase):\n    def test_sum_list_positive_numbers(self):\n        self.assertEqual(sum_list([1, 2, 3, 4, 5]), 15)\n\n    def test_sum_list_negative_numbers(self):\n        self.assertEqual(sum_list([-1, -2, -3, -4, -5]), -15)\n\n    def test_sum_list_mixed_numbers(self):\n        self.assertEqual(sum_list([1, -2, 3, -4, 5]), 3)\n\n    def test_sum_list_empty_list(self):\n        self.assertEqual(sum_list([]), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef square_elements(arr):\n    return [x ** 2 for x in arr]\n```",
        "completion": "```python\nimport unittest\n\nclass TestSquareElementsFunction(unittest.TestCase):\n    def test_square_elements_positive_numbers(self):\n        self.assertEqual(square_elements([1, 2, 3]), [1, 4, 9])\n\n    def test_square_elements_negative_numbers(self):\n        self.assertEqual(square_elements([-1, -2, -3]), [1, 4, 9])\n\n    def test_square_elements_mixed_numbers(self):\n        self.assertEqual(square_elements([1, -2, 3]), [1, 4, 9])\n\n    def test_square_elements_empty_list(self):\n        self.assertEqual(square_elements([]), [])\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef is_odd(n):\n    return n % 2 != 0\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsOddFunction(unittest.TestCase):\n    def test_is_odd_positive_number(self):\n        self.assertTrue(is_odd(3))\n\n    def test_is_odd_negative_number(self):\n        self.assertTrue(is_odd(-3))\n\n    def test_is_odd_even_number(self):\n        self.assertFalse(is_odd(4))\n\n    def test_is_odd_zero(self):\n        self.assertFalse(is_odd(0))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef string_length(s):\n    return len(s)\n```",
        "completion": "```python\nimport unittest\n\nclass TestStringLengthFunction(unittest.TestCase):\n    def test_string_length_regular(self):\n        self.assertEqual(string_length('hello'), 5)\n\n    def test_string_length_empty(self):\n        self.assertEqual(string_length(''), 0)\n\n    def test_string_length_spaces(self):\n        self.assertEqual(string_length(' '), 1)\n\n    def test_string_length_special_characters(self):\n        self.assertEqual(string_length('@#$%'), 4)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```",
        "completion": "```python\nimport unittest\n\nclass TestGCDFunction(unittest.TestCase):\n    def test_gcd_coprime_numbers(self):\n        self.assertEqual(gcd(7, 5), 1)\n\n    def test_gcd_small_numbers(self):\n        self.assertEqual(gcd(20, 8), 4)\n\n    def test_gcd_large_numbers(self):\n        self.assertEqual(gcd(100, 80), 20)\n\n    def test_gcd_one_zero(self):\n        self.assertEqual(gcd(0, 5), 5)\n        self.assertEqual(gcd(5, 0), 5)\n\n    def test_gcd_both_zeros(self):\n        self.assertEqual(gcd(0, 0), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef multiply(a, b):\n    return a * b\n```",
        "completion": "```python\nimport unittest\n\nclass TestMultiplyFunction(unittest.TestCase):\n    def test_multiply_positive_numbers(self):\n        self.assertEqual(multiply(2, 3), 6)\n\n    def test_multiply_negative_numbers(self):\n        self.assertEqual(multiply(-2, -3), 6)\n\n    def test_multiply_mixed_numbers(self):\n        self.assertEqual(multiply(2, -3), -6)\n\n    def test_multiply_zero(self):\n        self.assertEqual(multiply(0, 5), 0)\n        self.assertEqual(multiply(5, 0), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef capitalize(s):\n    return s.capitalize()\n```",
        "completion": "```python\nimport unittest\n\nclass TestCapitalizeFunction(unittest.TestCase):\n    def test_capitalize_lowercase(self):\n        self.assertEqual(capitalize('hello'), 'Hello')\n\n    def test_capitalize_uppercase(self):\n        self.assertEqual(capitalize('HELLO'), 'Hello')\n\n    def test_capitalize_empty_string(self):\n        self.assertEqual(capitalize(''), '')\n\n    def test_capitalize_mixed_case(self):\n        self.assertEqual(capitalize('hElLo'), 'Hello')\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef flip_boolean(value):\n    return not value\n```",
        "completion": "```python\nimport unittest\n\nclass TestFlipBooleanFunction(unittest.TestCase):\n    def test_flip_boolean_true(self):\n        self.assertFalse(flip_boolean(True))\n\n    def test_flip_boolean_false(self):\n        self.assertTrue(flip_boolean(False))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef factorial_iterative(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result\n```",
        "completion": "```python\nimport unittest\n\nclass TestFactorialIterativeFunction(unittest.TestCase):\n    def test_factorial_iterative_zero(self):\n        self.assertEqual(factorial_iterative(0), 1)\n\n    def test_factorial_iterative_one(self):\n        self.assertEqual(factorial_iterative(1), 1)\n\n    def test_factorial_iterative_positive(self):\n        self.assertEqual(factorial_iterative(5), 120)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef harmonic_mean(a, b):\n    return 2 / (1/a + 1/b)```",
        "completion": "```python\nimport unittest\n\nclass TestHarmonicMeanFunction(unittest.TestCase):\n    def test_harmonic_mean_positive_numbers(self):\n        self.assertAlmostEqual(harmonic_mean(2, 3), 2.4, places=7)\n\n    def test_harmonic_mean_negative_numbers(self):\n        with self.assertRaises(ValueError):\n            harmonic_mean(-2, 3)\n\n    def test_harmonic_mean_zero(self):\n        with self.assertRaises(ZeroDivisionError):\n            harmonic_mean(0, 5)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef is_anagram(s1, s2):\n    return sorted(s1) == sorted(s2)\n```",
        "completion": "```python\nimport unittest\n\nclass TestIsAnagramFunction(unittest.TestCase):\n    def test_is_anagram_true(self):\n        self.assertTrue(is_anagram('listen', 'silent'))\n\n    def test_is_anagram_false(self):\n        self.assertFalse(is_anagram('hello', 'world'))\n\n    def test_is_anagram_different_lengths(self):\n        self.assertFalse(is_anagram('hello', 'helloo'))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef merge_lists(lst1, lst2):\n    return sorted(lst1 + lst2)\n```",
        "completion": "```python\nimport unittest\n\nclass TestMergeListsFunction(unittest.TestCase):\n    def test_merge_lists_positive_numbers(self):\n        self.assertEqual(merge_lists([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6])\n\n    def test_merge_lists_negative_numbers(self):\n        self.assertEqual(merge_lists([-1, -3, -5], [-2, -4, -6]), [-6, -5, -4, -3, -2, -1])\n\n    def test_merge_lists_empty_one_list(self):\n        self.assertEqual(merge_lists([], [1, 2, 3]), [1, 2, 3])\n\n    def test_merge_lists_empty_both_lists(self):\n        self.assertEqual(merge_lists([], []), [])\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef distance(x1, y1, x2, y2):\n    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5\n```",
        "completion": "```python\nimport unittest\n\nclass TestDistanceFunction(unittest.TestCase):\n    def test_distance_origin(self):\n        self.assertAlmostEqual(distance(0, 0, 0, 0), 0, places=7)\n\n    def test_distance_positive_coordinates(self):\n        self.assertAlmostEqual(distance(0, 0, 3, 4), 5, places=7)\n\n    def test_distance_negative_coordinates(self):\n        self.assertAlmostEqual(distance(-1, -1, -4, -5), 5, places=7)\n\n    def test_distance_mixed_coordinates(self):\n        self.assertAlmostEqual(distance(1, 2, -2, -3), 5.830951894845301, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef add(a, b):\n    return a + b\n```",
        "completion": "```python\nimport unittest\n\nclass TestAddFunction(unittest.TestCase):\n    def test_add_positive_numbers(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_add_negative_numbers(self):\n        self.assertEqual(add(-1, -1), -2)\n\n    def test_add_mixed_numbers(self):\n        self.assertEqual(add(-1, 1), 0)\n\n    def test_add_zero(self):\n        self.assertEqual(add(0, 0), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef sqrt(x):\n    return x ** 0.5\n```",
        "completion": "```python\nimport unittest\n\nclass TestSqrtFunction(unittest.TestCase):\n    def test_sqrt_positive_number(self):\n        self.assertAlmostEqual(sqrt(4), 2, places=7)\n\n    def test_sqrt_zero(self):\n        self.assertAlmostEqual(sqrt(0), 0, places=7)\n\n    def test_sqrt_one(self):\n        self.assertAlmostEqual(sqrt(1), 1, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef min_value(arr):\n    return min(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestMinValueFunction(unittest.TestCase):\n    def test_min_value_positive_numbers(self):\n        self.assertEqual(min_value([1, 2, 3, 4, 5]), 1)\n\n    def test_min_value_negative_numbers(self):\n        self.assertEqual(min_value([-1, -2, -3, -4, -5]), -5)\n\n    def test_min_value_mixed_numbers(self):\n        self.assertEqual(min_value([1, -2, 3, -4, 5]), -4)\n\n    def test_min_value_single_element(self):\n        self.assertEqual(min_value([1]), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef concatenate_strings(str1, str2):\n    return str1 + str2\n```",
        "completion": "```python\nimport unittest\n\nclass TestConcatenateStringsFunction(unittest.TestCase):\n    def test_concatenate_two_non_empty_strings(self):\n        self.assertEqual(concatenate_strings('Hello', 'World'), 'HelloWorld')\n\n    def test_concatenate_empty_and_non_empty_string(self):\n        self.assertEqual(concatenate_strings('', 'World'), 'World')\n\n    def test_concatenate_two_empty_strings(self):\n        self.assertEqual(concatenate_strings('', ''), '')\n\n    def test_concatenate_space_and_string(self):\n        self.assertEqual(concatenate_strings(' ', 'World'), ' World')\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n```",
        "completion": "```python\nimport unittest\n\nclass TestDivideFunction(unittest.TestCase):\n    def test_divide_large_numbers(self):\n        self.assertAlmostEqual(divide(1e10, 1e2), 1e8)\n\n    def test_divide_by_float_numbers(self):\n        self.assertAlmostEqual(divide(0.1, 0.2), 0.5, places=7)\n\n    def test_divide_by_zero_float(self):\n        with self.assertRaises(ValueError):\n            divide(1.0, 0.0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef power(a, b):\n    return a ** b\n```",
        "completion": "```python\nimport unittest\n\nclass TestPowerFunction(unittest.TestCase):\n    def test_power_large_numbers(self):\n        self.assertEqual(power(10, 10), 1e10)\n\n    def test_power_zero_exponent(self):\n        self.assertEqual(power(2, 0), 1)\n\n    def test_power_negative_exponent(self):\n        self.assertAlmostEqual(power(2, -1), 0.5, places=7)\n\n    def test_power_float_exponent(self):\n        self.assertAlmostEqual(power(9, 0.5), 3, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
]

validation_data = [
    {
        "prompt": "```python\ndef find_max(arr):\n    return max(arr)\n```",
        "completion": "```python\nimport unittest\n\nclass TestFindMaxFunction(unittest.TestCase):\n    def test_find_max_positive_numbers(self):\n        self.assertEqual(find_max([1, 2, 3, 4, 5]), 5)\n\n    def test_find_max_negative_numbers(self):\n        self.assertEqual(find_max([-1, -2, -3, -4, -5]), -1)\n\n    def test_find_max_mixed_numbers(self):\n        self.assertEqual(find_max([1, -2, 3, -4, 5]), 5)\n\n    def test_find_max_single_element(self):\n        self.assertEqual(find_max([1]), 1)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "def is_odd(n):",
        "completion": "```python\nimport unittest\n\nclass TestIsOddFunction(unittest.TestCase):\n    def test_is_odd_positive_number(self):\n        self.assertTrue(is_odd(3))\n\n    def test_is_odd_negative_number(self):\n        self.assertTrue(is_odd(-3))\n\n    def test_is_odd_even_number(self):\n        self.assertFalse(is_odd(4))\n\n    def test_is_odd_zero(self):\n        self.assertFalse(is_odd(0))\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef distance(x1, y1, x2, y2):\n    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5\n```",
        "completion": "```python\nimport unittest\n\nclass TestDistanceFunction(unittest.TestCase):\n    def test_distance_origin(self):\n        self.assertAlmostEqual(distance(0, 0, 0, 0), 0, places=7)\n\n    def test_distance_positive_coordinates(self):\n        self.assertAlmostEqual(distance(0, 0, 3, 4), 5, places=7)\n\n    def test_distance_negative_coordinates(self):\n        self.assertAlmostEqual(distance(-1, -1, -4, -5), 5, places=7)\n\n    def test_distance_mixed_coordinates(self):\n        self.assertAlmostEqual(distance(1, 2, -2, -3), 5.830951894845301, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "def distance(x1, y1, x2, y2):    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5,",
        "completion": "```python\nimport unittest\n\nclass TestDistanceFunction(unittest.TestCase):\n    def test_distance_origin(self):\n        self.assertAlmostEqual(distance(0, 0, 0, 0), 0, places=7)\n\n    def test_distance_positive_coordinates(self):\n        self.assertAlmostEqual(distance(0, 0, 3, 4), 5, places=7)\n\n    def test_distance_negative_coordinates(self):\n        self.assertAlmostEqual(distance(-1, -1, -4, -5), 5, places=7)\n\n    def test_distance_mixed_coordinates(self):\n        self.assertAlmostEqual(distance(1, 2, -2, -3), 5.830951894845301, places=7)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
    {
        "prompt": "```python\ndef add(a, b):\n    return a + b\n```",
        "completion": "```python\nimport unittest\n\nclass TestAddFunction(unittest.TestCase):\n    def test_add_positive_numbers(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_add_negative_numbers(self):\n        self.assertEqual(add(-1, -1), -2)\n\n    def test_add_mixed_numbers(self):\n        self.assertEqual(add(-1, 1), 0)\n\n    def test_add_zero(self):\n        self.assertEqual(add(0, 0), 0)\n\nif __name__ == '__main__':\n    unittest.main()\n```"
    },
]

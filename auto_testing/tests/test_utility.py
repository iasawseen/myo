import unittest

from ..utils import utility


class FlattenTestCase(unittest.TestCase):
    def test_emtpy_iterable(self):
        lst = []
        result = utility.flatten(lst)
        self.assertEqual([], list(result))

    def test_not_nested_iterable(self):
        lst = list(range(10))
        result = utility.flatten(lst)
        self.assertEqual(lst, list(result))

    def test_single_nested_iterable(self):
        lst = [range(10), *range(10, 20)]
        result = utility.flatten(lst)
        self.assertEqual(list(range(20)), list(result))

    def test_multi_nested_iterable(self):
        lst = [[range(10), [range(10, 20)]], *range(20, 30), [range(30, 40)]]
        result = utility.flatten(lst)
        self.assertEqual(list(range(40)), list(result))


class IsFilePathValidWith4CodeTestCase(unittest.TestCase):
    def test_emtpy_valid_codes(self):
        file_path = '..\\raw_data\\00\\raw_00_0_0_0.pickle'
        result = utility.is_file_path_valid(file_path, 0)
        self.assertTrue(result)

    def test_true_with_one_valid_codes(self):
        valid_codes = ['00']
        file_path = '..\\raw_data\\00\\raw_00_0_0_0.pickle'
        result = utility.is_file_path_valid(file_path, 0, valid_codes)
        self.assertTrue(result)

    def test_false_with_one_valid_codes(self):
        valid_codes = ['01']
        file_path = '..\\raw_data\\00\\raw_00_0_0_0.pickle'
        result = utility.is_file_path_valid(file_path, 0, valid_codes)
        self.assertFalse(result)

    def test_true_with_multiple_valid_codes(self):
        valid_codes = ['00', '01', '02']
        file_path = '..\\raw_data\\00\\raw_00_0_0_0.pickle'
        result = utility.is_file_path_valid(file_path, 0, valid_codes)
        self.assertTrue(result)

    def test_false_with_multiple_valid_codes(self):
        valid_codes = ['000', '01', '02']
        file_path = '..\\raw_data\\00\\raw_00_0_0_0.pickle'
        result = utility.is_file_path_valid(file_path, 0, valid_codes)
        self.assertFalse(result)


class PipeTestCase(unittest.TestCase):
    def test_emtpy_funcs(self):
        init_data = 1
        result = utility.pipe(init_data)
        self.assertEqual(init_data, result)

    def test_not_emtpy_funcs(self):
        init_data = 1
        result = utility.pipe(init_data, [lambda x: x + 1, lambda x: x ** 2])
        self.assertEqual(4, result)

    def test_multiple_output_funcs(self):
        init_data = 1
        result = utility.pipe(init_data, [lambda x: (x, x + 1), lambda xy: xy[0] + xy[1]])
        self.assertEqual(3, result)

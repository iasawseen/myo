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



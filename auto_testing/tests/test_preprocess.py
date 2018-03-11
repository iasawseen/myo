import unittest

from ..processing import preprocess


class ClearTerminalNonesFromIterablesTestCase(unittest.TestCase):
    def test_clear_terminal_nones_from_empty_iterables(self):
        lst1 = []
        lst2 = []
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(0, len(result[0]))
        self.assertEqual(0, len(result[1]))

    def test_clear_terminal_nones_from_non_empty_iterables_without_nones(self):
        lst1 = list(range(10))
        lst2 = list(range(10, 20))
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(list(range(10)), result[0])
        self.assertEqual(list(range(10, 20)), result[1])

    def test_clear_terminal_nones_from_non_empty_iterables_with_equal_nones_on_left(self):
        lst1 = [None, *list(range(10))]
        lst2 = [None, *list(range(10, 20))]
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(list(range(10)), result[0])
        self.assertEqual(list(range(10, 20)), result[1])

    def test_clear_terminal_nones_from_non_empty_iterables_with_equal_nones_on_right(self):
        lst1 = [*list(range(10)), None]
        lst2 = [*list(range(10, 20)), None]
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(list(range(10)), result[0])
        self.assertEqual(list(range(10, 20)), result[1])

    def test_clear_terminal_nones_from_non_empty_iterables_with_not_equal_nones_on_left(self):
        lst1 = [None, None, None, *list(range(9))]
        lst2 = [None, None, *list(range(10, 20))]
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(list(range(9)), result[0])
        self.assertEqual(list(range(10, 20))[1:], result[1])

    def test_clear_terminal_nones_from_non_empty_iterables_with_not_equal_nones_on_right(self):
        lst1 = [*list(range(10)), None, None]
        lst2 = [*list(range(10, 21)), None]
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(list(range(10)), result[0])
        self.assertEqual(list(range(10, 20)), result[1])

    def test_clear_terminal_nones_from_non_empty_iterables_with_not_equal_nones_on_both_sides(self):
        lst1 = [None, None, *list(range(10)), None, None]
        lst2 = [None, None, None, *list(range(10, 20)), None]
        result = preprocess._clear_terminal_nones_from_iterables(lst1, lst2)
        self.assertEqual(2, len(result))
        self.assertEqual(list(range(1, 10)), result[0])
        self.assertEqual(list(range(10, 19)), result[1])


class TrimOnsideTestCase(unittest.TestCase):
    def test_trim_onside_left_no_deleting(self):
        lst1 = [-0.001, *list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='left')
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2, res2)

    def test_trim_onside_left_deleting(self):
        lst1 = [-0.02, *list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='left')
        self.assertEqual(lst1[1:], res1)
        self.assertEqual(lst2, res2)

    def test_trim_onside_left_multiple_deleting(self):
        lst1 = [-2, -0.1, -0.02, *list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='left')
        self.assertEqual(lst1[3:], res1)
        self.assertEqual(lst2, res2)

    def test_trim_onside_right_no_deleting(self):
        lst1 = [*list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='right')
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2, res2)

    def test_trim_onside_right_deleting(self):
        lst1 = [*list(range(10))]
        lst2 = [*list(range(10)), 9.1]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='right')
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2[:-1], res2)

    def test_trim_onside_right_multiple_deleting(self):
        lst1 = [*list(range(10))]
        lst2 = [*list(range(10)), 9.1, 10, 12, 14]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='right')
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2[:-4], res2)


class TrimToSyncTestCase(unittest.TestCase):
    def test_trim_to_sync_no_deleting(self):
        lst1 = [-0.001, *list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_onside(lst1, lst2, eps=0.01, side='left')
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2, res2)

    def test_trim_to_sync_left_deleting(self):
        lst1 = [-0.02, *list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_to_sync(lst1, lst2, eps=0.01)
        self.assertEqual(lst1[1:], res1)
        self.assertEqual(lst2, res2)

    def test_trim_to_sync_left_multiple_deleting(self):
        lst1 = [-2, -0.1, -0.02, *list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_to_sync(lst1, lst2, eps=0.01)
        self.assertEqual(lst1[3:], res1)
        self.assertEqual(lst2, res2)

    def test_trim_to_sync_right_no_deleting(self):
        lst1 = [*list(range(10))]
        lst2 = [*list(range(10)), 9.001]

        res1, res2 = preprocess._trim_to_sync(lst1, lst2, eps=0.01)
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2, res2)

    def test_trim_to_sync_right_deleting(self):
        lst1 = [*list(range(10))]
        lst2 = [*list(range(10)), 9.1]

        res1, res2 = preprocess._trim_to_sync(lst1, lst2, eps=0.01)
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2[:-1], res2)

    def test_trim_to_sync_right_multiple_deleting(self):
        lst1 = [*list(range(10))]
        lst2 = [*list(range(10)), 9.1, 10, 12, 14]

        res1, res2 = preprocess._trim_to_sync(lst1, lst2, eps=0.01)
        self.assertEqual(lst1, res1)
        self.assertEqual(lst2[:-4], res2)

    def test_trim_to_sync_both_sides_multiple_deleting(self):
        lst1 = [-2, -0.1, -0.02, *list(range(10))]
        lst2 = [*list(range(10)), 9.1, 10, 12, 14]

        res1, res2 = preprocess._trim_to_sync(lst1, lst2, eps=0.01)
        self.assertEqual(list(range(10)), res1)
        self.assertEqual(list(range(10)), res2)


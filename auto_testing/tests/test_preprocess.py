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


class GetConsecutiveNoneLengthsTestCase(unittest.TestCase):
    def test_emtpy_iterable(self):
        lst = []
        result = preprocess._get_consecutive_none_lengths(lst)
        self.assertEqual([], result)

    def test_no_nones_iterable(self):
        lst = list(range(10))
        result = preprocess._get_consecutive_none_lengths(lst)
        self.assertEqual([], result)

    def test_several_nones_iterable(self):
        lst = [None, None, 1, 2, None, None, None]
        result = preprocess._get_consecutive_none_lengths(lst)
        self.assertEqual([2, 3], result)

    def test_multiple_nones_iterable(self):
        lst = [None, None, 1, 2, 3, None, None, 5, None, None, None, None]
        result = preprocess._get_consecutive_none_lengths(lst)
        self.assertEqual([2, 2, 4], result)


class ValidateTestCase(unittest.TestCase):
    def test_emtpy_iterable(self):
        lst = []
        result = preprocess._validate(lst)
        self.assertEqual(True, result)

    def test_no_nones_iterable(self):
        lst = list(range(10))
        result = preprocess._validate(lst)
        self.assertEqual(True, result)

    def test_several_nones_iterable_max_limit2(self):
        lst = [None, None, 1, 2, None, None, None]
        result = preprocess._validate(lst, max_nones_limit=2, ave_nones_limit=10)
        self.assertEqual(False, result)

    def test_several_nones_iterable_max_limit3(self):
        lst = [None, None, 1, 2, None, None, None]
        result = preprocess._validate(lst, max_nones_limit=3, ave_nones_limit=10)
        self.assertEqual(True, result)

    def test_several_nones_iterable_max_ave2(self):
        lst = [None, None, 1, 2, None, None, None]
        result = preprocess._validate(lst, max_nones_limit=3, ave_nones_limit=2)
        self.assertEqual(False, result)

    def test_several_nones_iterable_max_ave3(self):
        lst = [None, None, 1, 2, None, None, None]
        result = preprocess._validate(lst, max_nones_limit=3, ave_nones_limit=3)
        self.assertEqual(True, result)


class CompressAnglesTestCase(unittest.TestCase):
    def test(self):
        angles = [[0] * 3,  # wrist
                  [[1] * 3, [2] * 3, [3] * 3],  # thumb
                  [[4] * 3, [5] * 3, [6] * 3],  # index
                  [[7] * 3, [8] * 3, [9] * 3],  # middle
                  [[10] * 3, [11] * 3, [12] * 3],  # ring
                  [[13] * 3, [14] * 3, [15] * 3]]  # pinky

        expected = [[0] * 3,
                    [1] * 3, 2, 3,
                    [4] * 3, 5, 6,  # index
                    [7] * 3, 8, 9,  # middle
                    [10] * 3, 11, 12,  # ring
                    [13] * 3, 14, 15]  # pinky

        result = preprocess._compress_angles(angles)
        self.assertEqual(expected, result)


class FillAnglesGapTestCase(unittest.TestCase):
    def test_single_gap(self):
        angles = [[0, 1, 0, 1],
                  None,
                  [2, 2, 2, 2]]

        expected = [[0, 1, 0, 1],
                    [1, 1.5, 1, 1.5],
                    [2, 2, 2, 2]]
        preprocess._fill_angles_gap(angles, 1, 2)
        self.assertEqual(expected, angles)

    def test_double_gap(self):
        angles = [[0, 1, 0, 1],
                  None,
                  None,
                  [3, 4, 3, 4]]

        expected = [[0, 1, 0, 1],
                    [1, 2, 1, 2],
                    [2, 3, 2, 3],
                    [3, 4, 3, 4]]

        preprocess._fill_angles_gap(angles, 1, 3)
        self.assertEqual(expected, angles)


class InterpolateAnglesTestCase(unittest.TestCase):
    def test_single_gap(self):
        angles = [[0, 1, 0, 1],
                  None,
                  [2, 2, 2, 2]]

        expected = [[0, 1, 0, 1],
                    [1, 1.5, 1, 1.5],
                    [2, 2, 2, 2]]
        preprocess._interpolate_angles(angles)
        self.assertEqual(expected, angles)

    def test_double_gap(self):
        angles = [[0, 1, 0, 1],
                  None,
                  None,
                  [3, 4, 3, 4]]

        expected = [[0, 1, 0, 1],
                    [1, 2, 1, 2],
                    [2, 3, 2, 3],
                    [3, 4, 3, 4]]

        preprocess._interpolate_angles(angles)
        self.assertEqual(expected, angles)

    def test_two_gaps(self):
        angles = [[0, 1, 0, 1],
                  None,
                  None,
                  [3, 4, 3, 4],
                  None,
                  [4, 5, 4, 5]]

        expected = [[0, 1, 0, 1],
                    [1, 2, 1, 2],
                    [2, 3, 2, 3],
                    [3, 4, 3, 4],
                    [3.5, 4.5, 3.5, 4.5],
                    [4, 5, 4, 5]]

        preprocess._interpolate_angles(angles)
        self.assertEqual(expected, angles)

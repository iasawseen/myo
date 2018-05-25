import unittest

from ..processing import alignment


class AlignmentTestCase(unittest.TestCase):

    def setUp(self):
        self.alignment = alignment.Alignment()

    def test_conventional_align_simple(self):
        self.alignment.align(list(range(1, 10)), [1, 4, 7, 9], eps=0.001)
        left, right = self.alignment.get_alignment(empty='-')
        left_str = '.'.join(str(el) for el in left)
        right_str = '.'.join(str(el) for el in right)
        self.assertEqual(left_str, '1.2.3.4.5.6.7.8.9')
        self.assertEqual(right_str, '1.-.-.4.-.-.7.-.9')

    def test_fast_align_against_conventional_simple(self):
        left_array = [1, 1, 2, 2, 2, 2, 2, 3, 4, 4, 5, 6, 7, 7]
        right_array = [1, 2, 2, 3, 4, 5, 7]

        left, right = self.alignment.align(left_array, right_array, 0.3)
        left_fast, right_fast = self.alignment.align_fast(left_array, right_array, 0.3, width=4)

        self.assertEqual(left, left_fast)
        self.assertEqual(right, right_fast)

    def test_fast_align_against_conventional_medium(self):
        left_array = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8]
        right_array = [1, 2, 2, 3, 4, 5, 7, 8]

        left, right = self.alignment.align_fast(left_array, right_array, 0.3, width=4)
        left_fast, right_fast = self.alignment.align_fast(left_array, right_array, 0.3, width=4)

        self.assertEqual(left, left_fast)
        self.assertEqual(right, right_fast)

    def test_fast_align_unbalanced(self):
        left_array = [1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8]
        right_array = [1, 4, 7, 8]

        left, right = self.alignment.align_fast(left_array, right_array, 0.3, width=4)
        left_fast, right_fast = self.alignment.align_fast(left_array, right_array, 0.3, width=4)

        self.assertEqual(left, left_fast)
        self.assertEqual(right, right_fast)

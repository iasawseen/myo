import numpy as np
import unittest

from ..processing import transformers


class AverageFilterTestCase(unittest.TestCase):
    def test_one_column_window1(self):
        array = np.array([[1],
                          [3],
                          [5]])
        filtered_array = transformers.average_filter(array, window=1)
        self.assertTrue(np.allclose(array, filtered_array))

    def test_two_columns_window1(self):
        array = np.array([[1, 2],
                          [3, 4],
                          [5, 6]])
        filtered_array = transformers.average_filter(array, window=1)
        self.assertTrue(np.allclose(array, filtered_array))

    def test_one_column_window2(self):
        array = np.array([[1],
                          [3],
                          [5]])
        filtered_array = transformers.average_filter(array, window=2)
        expected_array = np.array([[0.5],
                                   [2],
                                   [4]])
        self.assertTrue(np.allclose(expected_array, filtered_array))

    def test_one_column_window3(self):
        array = np.array([[1],
                          [3],
                          [5]])
        filtered_array = transformers.average_filter(array, window=3)
        expected_array = np.array([[4/3],
                                   [3],
                                   [8/3]])
        self.assertTrue(np.allclose(expected_array, filtered_array))

    def test_two_columns_window3(self):
        array = np.array([[1, 2],
                          [3, 4],
                          [5, 6]])
        filtered_array = transformers.average_filter(array, window=3)
        expected_array = np.array([[4/3, 6/3],
                                   [3, 4],
                                   [8/3, 10/3]])
        self.assertTrue(np.allclose(expected_array, filtered_array))


class AddWindowToXYTestCase(unittest.TestCase):
    def test_one_column_window1(self):
        x = np.array([[1],
                      [3],
                      [5],
                      [7],
                      [9]])
        y = np.array([[0],
                      [2],
                      [4],
                      [6],
                      [8]])

        x_windowed, y_windowed = transformers.add_window_to_xy((x, y), window=1)

        x_expected = np.array([[[1]],
                               [[3]],
                               [[5]],
                               [[7]]])

        equal = np.allclose(x_expected, x_windowed) and np.allclose(y[:-1, :], y_windowed)
        self.assertTrue(equal)

    def test_one_column_window2(self):
        x = np.array([[1],
                      [3],
                      [5],
                      [7],
                      [9]])

        y = np.array([[0],
                      [2],
                      [4],
                      [6],
                      [8]])

        x_windowed, y_windowed = transformers.add_window_to_xy((x, y), window=2)

        x_expected = np.array([[[1],
                                [3]],
                               [[3],
                                [5]],
                               [[5],
                                [7]]])

        y_expected = np.array([[2],
                               [4],
                               [6]])

        equal = np.allclose(x_expected, x_windowed) and np.allclose(y_expected, y_windowed)
        self.assertTrue(equal)

    def test_multiple_column_window2(self):
        x = np.array([[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8],
                      [9, 10]])

        y = np.array([[0, 1, 2],
                      [2, 3, 4],
                      [4, 5, 6],
                      [6, 7, 8],
                      [8, 9, 10]])

        x_windowed, y_windowed = transformers.add_window_to_xy((x, y), window=2)

        x_expected = np.array([[[1, 2],
                                [3, 4]],
                               [[3, 4],
                                [5, 6]],
                               [[5, 6],
                                [7, 8]]])

        y_expected = np.array([[2, 3, 4],
                               [4, 5, 6],
                               [6, 7, 8]])

        equal = np.allclose(x_expected, x_windowed) and np.allclose(y_expected, y_windowed)
        self.assertTrue(equal)


class CompactXYTestCase(unittest.TestCase):
    def test_one_column_ratio2(self):
        x = np.array([[1],
                      [3],
                      [5],
                      [7],
                      [9]])
        y = np.array([[0],
                      [2],
                      [4],
                      [6],
                      [8]])

        x_compacted, y_compacted = transformers.compact_xy((x, y), ratio=2)

        x_expected = np.array([[1],
                               [5],
                               [9]])
        y_expected = np.array([[0],
                               [4],
                               [8]])

        equal = np.allclose(x_expected, x_compacted) and np.allclose(y_expected, y_compacted)
        self.assertTrue(equal)

    def test_one_column_ratio3(self):
        x = np.array([[1],
                      [3],
                      [5],
                      [7],
                      [9]])
        y = np.array([[0],
                      [2],
                      [4],
                      [6],
                      [8]])

        x_compacted, y_compacted = transformers.compact_xy((x, y), ratio=3)

        x_expected = np.array([[1],
                               [7]])
        y_expected = np.array([[0],
                               [6]])

        equal = np.allclose(x_expected, x_compacted) and np.allclose(y_expected, y_compacted)
        self.assertTrue(equal)

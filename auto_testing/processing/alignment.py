from __future__ import print_function

import numpy as np
import unittest
import time


class Alignment:
    def __init__(self):
        self.left = None
        self.right = None
        self.max_indel = None
        self.n = None
        self.m = None
        self.matrix = None
        self.backtrack = None
        self.delete_length = None
        self.insert_length = None
        self.j_offset = None
        self.fn = None

    def align(self, left, right, eps, access_fn=lambda x, i: x[i]):
        self.left = left
        self.right = right
        self.fn = access_fn

        self.n = len(self.left) + 1
        self.m = len(self.right) + 1

        self.matrix = np.zeros((self.n, self.m))
        self.backtrack = np.zeros((self.n, self.m))

        self.delete_length = np.zeros((self.n, self.m))

        self.matrix[0, range(1, self.m)] = [self.matrix[0, i - 1] - pow(1.3, i) for i in range(1, self.m)]
        self.matrix[range(1, self.n), 0] = -np.inf

        self.backtrack.fill(-1)

        for i in range(1, self.n):
            for j in range(1, self.m):
                subs_score = 3 if abs(self.fn(self.left, i - 1) - self.fn(self.right, j - 1)) < eps else -3

                match = self.matrix[i - 1, j - 1] + subs_score
                delete = self.matrix[i - 1, j] - pow(1.3, self.delete_length[i - 1, j])

                actions = np.array([match, delete])
                argmax = np.argmax(actions)

                self.matrix[i, j] = actions[argmax]
                self.backtrack[i, j] = argmax

                if argmax == 1:
                    self.delete_length[i, j] = self.delete_length[i - 1, j] + 1

        return self.get_alignment()

    def align_fast(self, left, right, eps, access_fn=lambda x, i: x[i], width=10):
        self.left = left
        self.right = right
        self.fn = access_fn

        self.n = len(self.left) + 1
        self.m = width + 1

        self.matrix = np.zeros((self.n, self.m))
        self.backtrack = np.zeros((self.n, self.m))

        self.delete_length = np.zeros((self.n, self.m))
        self.j_offset = np.zeros(self.n, dtype=np.int64)

        self.matrix[0, range(1, self.m)] = [self.matrix[0, i - 1] - pow(1.3, i) for i in range(1, self.m)]
        self.matrix[range(1, self.n), 0] = -np.inf

        self.backtrack.fill(-1)

        m_coef = (len(right) + 1 - self.m) / float(self.n - 2 * self.m)
        self.j_offset[self.m: self.n - self.m] = [min(int(m_coef * i), len(right) + 1 - self.m)
                                                  for i in range(self.n - 2 * self.m)]

        self.j_offset[self.n - self.m:] = len(right) + 1 - self.m

        for i in range(1, self.n):
            for j in range(1, self.m):
                subs_score = 3 if abs(self.fn(self.left, i - 1) -
                                      self.fn(self.right, j - 1 + self.j_offset[i])) < eps else -3

                row_offset = self.j_offset[i] - self.j_offset[i - 1]

                match = self.matrix[i - 1, j - 1 + row_offset] + subs_score
                if j + row_offset >= self.m:
                    delete = -np.inf
                else:
                    delete = self.matrix[i - 1, j + row_offset] - pow(1.3, self.delete_length[i - 1, j + row_offset])

                actions = np.array([match, delete])
                argmax = np.argmax(actions)

                self.matrix[i, j] = actions[argmax]
                self.backtrack[i, j] = argmax

                if argmax == 1:
                    self.delete_length[i, j] = self.delete_length[i - 1, j + row_offset] + 1

        return self._get_alignment_fast()

    def get_alignment(self, empty=None):
        alignment_left = list()
        alignment_right = list()

        i = self.n - 1
        j = self.m - 1

        while i > 0 and j > 0:
            if self.backtrack[i, j] == 0:
                alignment_left.insert(0, self.left[i - 1])
                alignment_right.insert(0, self.right[j - 1])
                i -= 1
                j -= 1
            elif self.backtrack[i, j] == 1:
                alignment_left.insert(0, self.left[i - 1])
                alignment_right.insert(0, empty * len(str(self.left[i - 1])) if empty else empty)
                i -= 1
            else:
                raise ValueError('Wrong action in backtrack')

        return alignment_left, alignment_right

    def _get_alignment_fast(self, empty=None):
        alignment_left = list()
        alignment_right = list()

        i = self.n - 1
        j = self.m - 1

        while i > 0 and j > 0:
            row_offset = self.j_offset[i] - self.j_offset[i - 1]

            if self.backtrack[i, j] == 0:
                alignment_left.insert(0, self.left[i - 1])
                alignment_right.insert(0, self.right[j - 1 + self.j_offset[i]])
                i -= 1
                j = j - 1 + row_offset
            elif self.backtrack[i, j] == 1:
                alignment_left.insert(0, self.left[i - 1])
                alignment_right.insert(0, empty * len(str(self.left[i - 1])) if empty else empty)
                i -= 1
                j = j + row_offset
            else:
                raise ValueError('Wrong action in backtrack')

        return alignment_left, alignment_right

import collections
import numpy as np


def flatten(iterable):
    for item in iterable:
        if not isinstance(item, collections.Iterable):
            yield item
        else:
            for sub_item in flatten(item):
                yield sub_item




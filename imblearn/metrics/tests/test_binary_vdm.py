import numpy as np
from numpy.testing import (assert_array_equal, assert_equal)
import unittest

from imblearn.metrics.binary_vdm import BinaryVDM

def gen_data():
  X = np.array([[0., 1., 0.],
                [1., 0., 1.],
                [1., 0., 0.],
                [1., 1., 0.]])
  y = np.array([2, 3, 2, 3])
  return X, y

class TestBinaryVDM(unittest.TestCase):
  def test_fit(self):
    X, y = gen_data()
    vdm = BinaryVDM()
    vdm.fit(X,y)
    expected_total_value_stats = {0: np.array([1, 2, 3]),
                                  1: np.array([3, 2, 1])}
    expected_class_value_stats = {2: {0: np.array([1, 1, 2]),
                                      1: np.array([1, 1, 0])},
                                  3: {0: np.array([0, 1, 1]),
                                      1: np.array([2, 1, 1])}}

    for k in expected_total_value_stats.iterkeys():
      assert_array_equal(vdm.total_value_stats[k],
                         expected_total_value_stats[k])
    for k in vdm.total_value_stats.iterkeys():
      assert_array_equal(vdm.total_value_stats[k],
                         expected_total_value_stats[k])

    for c, v in expected_class_value_stats.iteritems():
      for inner_k in v.iterkeys():
        assert_array_equal(vdm.class_value_stats[c][inner_k],
                           v[inner_k])
    for c, v in vdm.class_value_stats.iteritems():
      for inner_k in v.iterkeys():
        assert_array_equal(v[inner_k],
                           expected_class_value_stats[c][inner_k])


if __name__ == '__main__':
  unittest.main()
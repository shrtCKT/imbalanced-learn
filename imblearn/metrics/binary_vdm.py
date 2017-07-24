""" Implementation of version of Value Difference Metric used by SMOTE-N
"""
import numpy as np

class BinaryVDM(object):
  def fit(self, X, y):
    """ Calculate VDM stats.
    
    :param X: A matrix of feature vector of zeros and ones.
    :param y: Class labels
    :return: 
    """
    self.classes = np.unique(y)
    self.class_value_stats = {}  # map class class to value stats
    self.total_value_stats = {0: np.zeros(X.shape[1], dtype=int),
                         1:np.zeros(X.shape[1], dtype=int)}
    for c in self.classes:
      X_class = X[y == c]
      self.class_value_stats[c] = {}
      self.class_value_stats[c][1] = X_class.sum(axis=0).astype(int)
      self.class_value_stats[c][0] = np.abs(X_class.shape[0] - self.class_value_stats[c][1])
      self.total_value_stats[1] += self.class_value_stats[c][1]
      self.total_value_stats[0] += self.class_value_stats[c][0]


  def pairwise(self, x1, x2):
    """ Calculate the BinaryVDM distance between two feature vectors.
    
    :param x1: feature vector 1
    :param x2: feature vector 2
    :return: the Binary VDM distance.
    """
    dist = 0.
    for j in xrange(len(x1)):
      dist += self._distance(x1[j], x2[j], j)

    return dist

  def _distance(self, x1j, x2j, j):
    dist = 0.
    x1j = 1 if x1j > 0.5 else 0   # for kd tree cases
    x2j = 1 if x2j > 0.5 else 0   # for kd tree cases
    for c in self.classes:
      dist = (self.class_value_stats[c][x1j][j] /
              float(self.total_value_stats[x1j][j])) - \
             (self.class_value_stats[c][x2j][j] /
              float(self.total_value_stats[x2j][j]))

    return dist

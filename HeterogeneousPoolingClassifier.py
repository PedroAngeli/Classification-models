import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.utils import resample

class HeterogeneousPoolingClassifier(BaseEstimator):
  def __init__(self, n_samples=1):
    super().__init__()
    self.n_samples = n_samples
    self.predictors = []

  def fit(self, x, y):
    classes, classes_frequency = np.unique(y, return_counts=True)
    self.classes_frequency = [0] * (len(classes))
    for i in range(len(classes)):
      self.classes_frequency[classes[i]] = classes_frequency[i]
    for i in range(self.n_samples):
      if i == 0:
        x_test, y_test = x, y
      else:
        x_test, y_test = resample(x, y, random_state=i-1)
      self.predictors.append(DecisionTreeClassifier().fit(x_test, y_test))
      self.predictors.append(GaussianNB().fit(x_test, y_test))
      self.predictors.append(KNeighborsClassifier().fit(x_test, y_test))

  def predict(self, x):
    results = np.array([predictor.predict(x) for predictor in self.predictors]).transpose()
    y_pred = []
    for result in results:
      classes_predicted, classes_frequency = np.unique(result, return_counts=True)
      most_voted_frequency = np.amax(classes_frequency)
      most_voted = [c for (c, f) in zip(classes_predicted, classes_frequency) if f == most_voted_frequency]
      most_voted.sort(key=lambda x:-self.classes_frequency[x])
      y_pred.append(most_voted[0])
    
    return np.array(y_pred)

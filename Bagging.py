import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier

class Bagging:
  def __init__(self): 
    self.bagging = BaggingClassifier() #modelo
    self.scalar = StandardScaler() #z-score
    self.grid = {'estimator__n_estimators': [10, 25, 50, 100]} #hiperparametros do grid search
    self.pipeline = Pipeline([('transformer', self.scalar), ('estimator', self.bagging)]) #pipeline de transformações
    self.gs = GridSearchCV(estimator=self.pipeline, param_grid = self.grid, scoring='accuracy', cv = 4) # definindo grid search
    self.rfk = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234) #Estratégia de validação cruzada

  def fit(self, x, y): #Treina o modelo
    print('Treinando Bagging...')
    self.scores = cross_val_score(self.gs, x, y, scoring='accuracy', cv = self.rfk)
    print('Finalizado!')
  
  def results(self): #Retorna a média, desvio padrão e intervalo de confiança
    if hasattr(self, 'scores') == False:
      print('You have to fit the model first.')
      return
    mean = self.scores.mean()
    std = self.scores.std()
    inf, sup = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(len(self.scores)))
    return (mean, std, inf, sup)

  def getScores(self):
    if hasattr(self, 'scores') == False:
      print('You have to fit the model first.')
      return
    return self.scores
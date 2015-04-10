#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
 
class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([
            ('imputer', Imputer(strategy='most_frequent')),
            ('sgd', SGDClassifier( loss = 'modified_huber', penalty = 'elasticnet', alpha = 1.4728334788264625, l1_ratio = 0.34555431482496163 ))
        ])
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
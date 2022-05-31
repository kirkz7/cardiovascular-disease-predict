import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
class DecisionTree:
    model = DecisionTreeClassifier()
    def baseline(self):
        self.model = DecisionTreeClassifier()
        print(self.model.get_params())
    def best_model(self,Xtrain,ytrain):
        criterion = ['gini', 'entropy']
        max_depth = [2, 4, 6, 8, 10, 12]
        model = DecisionTreeClassifier()
        parameters = dict({'criterion':criterion,'max_depth':max_depth})
        clf_GS = GridSearchCV(model, parameters)
        clf_GS.fit(Xtrain,ytrain)
        criterion = clf_GS.best_params_['criterion']
        max_depth = clf_GS.best_params_['max_depth']
        self.model = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
        print("Set the model with params:", clf_GS.best_params_)
    def fit(self,Xtrain,ytrain):
        self.model.fit(Xtrain,ytrain)

    def predict(self,Xtest):
        ypr = self.model.predict(Xtest)
        probTest = self.model.predict_proba(Xtest)
        probTest = probTest[:, 1]
        return ypr,probTest
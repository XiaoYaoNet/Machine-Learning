import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c

from sklearn import metrics

def Logist(path):
	data = pd.read_excel(path)
	data.dropna(inplace=True)
	array=data.values
	X=array[:,1:len(data.columns)-1]
	y=array[:,len(data.columns)-1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

	cls = LogisticRegression(C=1.0,tol=1e-6)
	rbf=cls.fit(X_train, y_train)
	y_pred = cls.predict(X_test)

	return (X_test,y_pred)
	
x,y=Logist("./test.xls")
print(x,y)
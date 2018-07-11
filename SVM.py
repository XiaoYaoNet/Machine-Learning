import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

#支持向量机
def SVM(path):
	data = pd.read_excel(path)
	data.dropna(inplace=True)
	array=data.values
	X=array[:,1:len(data.columns)-1]
	y=array[:,len(data.columns)-1]
	#X=preprocessing.scale(X)
	#y=preprocessing.scale(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

	rbf_svc = svm.SVR(kernel='rbf')
	rbf_svc.tol=1
	rbf=rbf_svc.fit(X_train, y_train)
	y_pred = rbf_svc.predict(X_test)

	return (X_test,y_pred)
	
x,y=SVM("./test.xls")
print(x,y)
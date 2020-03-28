from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import _pickle as pickle
from sklearn.externals import joblib
import numpy as np 

iris = datasets.load_iris()
X=iris.data
y=iris.target

print(X[0])

X_train,X_test,y_train,y_test = train_test_split(X,y)

rfc = RandomForestClassifier(n_estimators = 100, n_jobs = 2)
rfc.fit(X_train,y_train)
# print(rfc) 
# use pickle
# pickle.dump(rfc, open("iris_rfc.pkl","wb"))
joblib_file = "joblib_RL_Model.pkl"  
joblib.dump(rfc, joblib_file)

#Load 
# my_random_forest = pickle.load(open("iris_rfc.pkl","rb"))
# print(my_random_forest)



# print(my_random_forest.predict([[5.84,3.0,3.75,1.1]]))
# In[1]:
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm

data = fetch_olivetti_faces()
y = data.target


X_final = []
for i in range(len(data.images)):
    X_final.append(data.images[i].flatten())
X_final = np.array(X_final)


X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)



# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=150)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# # Data after PCA are not normalized. Re-apply normalization
# X_train = scaler.fit_transform(X_train)
# X_test= scaler.fit_transform(X_test)


clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
y_results = clf.predict(X_test)

correct = 0
for i in range(len(X_test)):
    if y_results[i] == y_test[i]:
        correct+=1
print("accuracy")
print(correct/len(X_test))

print("right and wrong classification examples")
print(y_results == y_test)
print("right classification example")
print(y_results[1] == y_test[1])
print("wrong classification example")
print(y_results[37] == y_test[37])

# In[2]:

from sklearn.model_selection import GridSearchCV
from time import time
from sklearn import svm
from sklearn import metrics
t01 = time()
print("change kernel parameters")

parameters = {'kernel':('linear', 'rbf','poly'),'C':(1,1e3, 1e4, 1e5)}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print("done in %0.3fs" % (time() - t01))
print(clf.cv_results_['mean_test_score'])
print("best parameters")
print(clf.best_params_)

print("accuracy for c = 1 , linear kernel")

t02 = time()
clf = svm.SVC(C=1, kernel='linear')
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t02))


print("Prediction")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))


t02 = time()
print("Change parameters for rbf kernel")
param_grid = {'C': [1,1e3, 1e4, 1e5],
              'gamma': [0.0001, 0.001, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t02))
print(clf.cv_results_['mean_test_score'])
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print(clf.best_score_)

print("Accuracy for  C =1000 , gamma = 0.0001 kernel = rbf")

clf = svm.SVC(C=1000,gamma= 0.0001, kernel='rbf')
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t02))


print("Prediction")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))


t02 = time()
print("Change parameters polynomial kernel")
param_grid = {'C': [1,1e3, 1e4, 1e5],
              'gamma': [0.0001, 0.001, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='poly', degree = 2, class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train, y_train)

print("done in %0.3fs" % (time() - t02))
print(clf.cv_results_['mean_test_score'])
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Accuracy for gamma= 0.01, C= 1 polynomial Kernel")
t01 = time()
svm = SVC(kernel='poly', random_state=None, gamma=0.01, C=1, degree=2)
svm.fit(X_train, y_train)
z_results = clf.predict(X_test)
print("done in %0.3fs" % (time() - t02))

from sklearn import metrics
print("Prediction")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

# Model Accuracy: how often is the classifier correct?
print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))



from time import time
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics

cValue = {1, 1e3, 1e4, 1e5}
kernelType = {'linear', 'rbf', 'poly'}
A = []
for theKernel in kernelType:
    for c in cValue:
        t01 = time()
        parametersA = {'kernel': [theKernel], 'C': [c]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(svc, parametersA, cv=5)
        clf.fit(X_train, y_train)
        print("training time")
        print("done in %0.3fs for %s/%s" % (time() - t01, theKernel, c))
        t0 = time()
        y_pred = clf.predict(X_test)
        print("testing time")
        print("done in %0.3fs for %s/%s" % (time() - t0, theKernel, c))

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
        A.append(metrics.accuracy_score(y_test, y_pred))
print("best parameters")
print(clf.best_params_)
print(" Best accuracy")
print(max(A)) 

cValue = {1, 1e3, 1e4,1e5}
gammaValue = {0.0001, 0.001, 0.01, 0.1}
B = []
for gamma in gammaValue:
    for c in cValue:
        t01 = time()
        parametersA = {'gamma': [gamma], 'C': [c]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                           parametersA, cv=5, iid=False)
        print("training time")
        clf = clf.fit(X_train, y_train)
        print("done in %0.3fs for %s/%s" % (time() - t01, gamma, c))
        t0 = time()
        y_pred = clf.predict(X_test)
        print("testing time")
        print("done in %0.3fs for %s/%s" % (time() - t0, gamma, c))

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
        B.append(metrics.accuracy_score(y_test, y_pred))
print("best parameters")
print(clf.best_params_)
print("Best accuracy")
print(max(B)) 


cValue = {1, 1e3, 1e4,1e5}
gammaValue = {0.0001,0.001, 0.01, 0.1}
C = []
for gamma in gammaValue:
    for c in cValue:
        t01 = time()
        parametersA = {'gamma': [gamma], 'C': [c]}
        svc = svm.SVC(gamma="scale")
        clf = GridSearchCV(SVC(kernel='poly', degree=2,),
                           parametersA, cv=5, iid=False)
        clf = clf.fit(X_train, y_train)
        print("training time")
        print("done in %0.3fs for %s/%s" % (time() - t01, gamma, c))
        t0 = time()
        y_pred = clf.predict(X_test)
        print("testing time")
        print("done in %0.3fs for %s/%s" % (time() - t0, gamma, c))

        # Model Accuracy: how often is the classifier correct?
        print("Accuracy score:", metrics.accuracy_score(y_test, y_pred))
        C.append(metrics.accuracy_score(y_test, y_pred))

print("best parameters")
print(clf.best_params_)
print("Best accuracy")
print(max(C)) 

print(" the best accurracy of all")
print(max(max(A),max(B),max(C)))


# In[4]
print("KNN Classification")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import time
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)

t0 = time()
print("Prediction")
y_pred =(neigh.predict(X_test))
print("done in %0.3fs" % (time() - t0))
print("Accuracy score:")
print(accuracy_score(y_test,y_pred))

print("Near Centroid Classification")
from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)

print("Prediction")
t0 = time()
NearestCentroid(metric='euclidean', shrink_threshold=None)
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print("Accuracy score:")
print(accuracy_score(y_test,y_pred))



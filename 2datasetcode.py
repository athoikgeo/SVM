# In[1]
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn import svm
import numpy as np
digits = load_digits()
print(digits.data.shape)
data = digits.data
y = digits.target


X_final = []
for i in range(len(digits.images)):
    X_final.append(digits.images[i].flatten())
X_final = np.array(X_final)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)


print("Tαξινόμηση των δεδομένων και υπολογισμός της απόδοσης της ταξινόμησης")

clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
y_results = clf.predict(X_test)

correct = 0
for i in range(len(X_test)):
    if y_results[i] == y_test[i]:
        correct+=1
print(correct/len(X_test))
print("Παραδείγματα ορθής και εσφαλμένης κατηγοριοποίησης")
print(y_results == y_test)
print("ορθή κατηγοριοποίηση")
print(y_results[2] == y_test[2])
# In[2]:
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn import svm
from sklearn import metrics
t01 = time()
print("Αλλαγή παραμέτρων συνάρτησης πυρήνα")

parameters = {'kernel':('linear', 'rbf','poly'),'C':(1,1e3, 1e4, 1e5)}
svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print("done in %0.3fs" % (time() - t01))
print(clf.cv_results_['mean_test_score'])
print("best parameters")
print(clf.best_params_)

print("Υπολογισμός της απόδοσης για c = 1 , linear kernel")

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
print("Αλλαγή παραμέτρων για rbf kernel")
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

print("Yολογισμός της απόδοσης για  C =1000 , gamma = 0.0001 kernel = rbf")

clf = svm.SVC(C=1000,gamma= 0.0001, kernel='rbf')
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t02))


print("Prediction")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))
print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))


t02 = time()
print("Αλλαγή παραμέτρων για polynomial kernel")
param_grid = {'C': [1,1e3, 1e4, 1e5],
              'gamma': [0.0001, 0.001, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='poly', degree = 2, class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train, y_train)

print("done in %0.3fs" % (time() - t02))
print(clf.cv_results_['mean_test_score'])
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Yπολογισμός της απόδοσης για gamma= 0.01, C= 1 polynomial Kernel")
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


print("Accuracy score:",metrics.accuracy_score(y_test, y_pred))


# In[3]
# O παρακάτω κώδικας υπολογίζει ξεχωριστά για την κάθε συνάρτηση πυρήνα
#  και την κάθε παράμετρο γ και C που χρησιμοποιήθηκαν την απόδοση του καθώς και
# τον χρόνο εκπαίδευσης και πρόβλεψης
print(" Υπολογισμός της απόδοσης και του χρόνου εκπαίδευσης και πρόβλεψης για κάθε παραπάνω συνδιασμό παραμετρων ")
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
print(max(A)) # η μέγιστη απόδοση

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
print(max(B)) # η μέγιστη απόδοση


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
print(max(C)) # η μέγιστη απόδοση

print(" the best accurracy of all")
print(max(max(A),max(B),max(C))) # η καλύτερη από όλες τις αποδόσεις


# In[4]
print("Ταξινόμηση βάσει πλησιέστερου γείτονα")
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

print("Ταξινόμηση βάσει πλησιέστερου κέντρου")
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


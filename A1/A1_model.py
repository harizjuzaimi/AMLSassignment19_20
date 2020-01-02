from sklearn import svm, datasets
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def SVM_A1(training_images, training_labels, test_images, test_labels):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                        ]


    classifier = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)  # gridsearchCV use 3 k-fold by default
    classifier.fit(training_images, training_labels)

# default score of SVM is accuracy
    acc_A1_train = classifier.best_score_

# prediction using best classifier choose by GridSearchCV
    pred = classifier.best_estimator_.predict(test_images)

    acc_A1_test = accuracy_score(test_labels, pred)

    return acc_A1_train, acc_A1_test




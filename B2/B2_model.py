from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


def SVM_B2(training_images, training_labels):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                        ]

    classifierB2 = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)  # gridsearchCV use 3 k-fold by default
    classifierB2.fit(training_images, training_labels)

    # default score of SVM is accuracy
    acc_B2_train = classifierB2.best_score_



    # prediction using best classifier choose by GridSearchCV
    classifierB2_tuned = classifierB2.best_estimator_

    return acc_B2_train, classifierB2_tuned
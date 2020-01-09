from sklearn import svm, datasets
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV



def SVM_A1(training_images, training_labels):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                        ]


    classifierA1 = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)  # gridsearchCV use 3 k-fold by default
    classifierA1.fit(training_images, training_labels)

# default score of SVM is accuracy
    acc_A1_train = classifierA1.best_score_

# prediction using best classifier choose by GridSearchCV
    classifierA1_tuned = classifierA1.best_estimator_


    return acc_A1_train, classifierA1_tuned




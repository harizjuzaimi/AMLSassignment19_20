from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def train(training_images, training_labels):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
                        {'kernel': ['poly'], 'degree': [2, 3], 'C': [1, 10, 100]}
                        ]

    classifierB1 = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)  # gridsearchCV use 3 k-fold by default
    classifierB1.fit(training_images, training_labels)

    # default score of SVM is accuracy
    acc_B1_train = classifierB1.best_score_

    # prediction using best classifier choose by GridSearchCV
    classifierB1_tuned = classifierB1.best_estimator_

    return acc_B1_train, classifierB1_tuned


def test(classifier, test_images, test_label):

    pred_B1 = classifier.predict(test_images)

    acc_B1 = accuracy_score(test_label, pred_B1)

    return acc_B1
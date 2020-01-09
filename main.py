import os
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler

# ======================================================================================================================
# Data preprocessing

from data_preprocessing import get_data_A1, get_data_A2, get_data_B1, get_data_B2
tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1 = get_data_A1()


# tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = get_data_A2()
#
#
# tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1 = get_data_B1()
#
#
# tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = get_data_B2()

# ======================================================================================================================
# Task A1
from A1 import A1_model
model_A1 = A1_model.SVM_A1(tr_X_A1, tr_Y_A1)
acc_A1_train, classifierA1_tuned = model_A1
predA1 = classifierA1_tuned.predict(te_X_A1)
acc_A1_test = accuracy_score(te_Y_A1, predA1)

from A1 import A1_landmarks
X_A1, y_A1 = A1_landmarks.extract_features_labels()
Y_A1 = np.array([y_A1, -(y_A1 - 1)]).T
temp = X_A1.reshape(len(X_A1), 68*2)
scaler = StandardScaler()
temp_X_A1 = scaler.fit_transform(temp)
temp_Y_A1 = list(zip(*Y_A1))[0]
pred_A1_add = classifierA1_tuned.predict(temp_X_A1)
acc_A1_add_test = accuracy_score(temp_Y_A1, pred_A1_add)

print(acc_A1_train, acc_A1_test, acc_A1_add_test)


# model_A1 = A1(args...)                 # Build model object.
# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
# from A2 import A2_model
# model_A2 = A2_model.SVM_A2(tr_X_A2, tr_Y_A2)
# acc_A2_train, classifierA2_tuned = model_A2
# pred_A2 = classifierA2_tuned.predict(te_X_A2)
# acc_A2_test = accuracy_score(te_Y_A2, pred_A2)



# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
# from B1 import B1_model
# model_B1 = B1_model.SVM_B1(tr_X_B1, tr_Y_B1)
# acc_B1_train, classifierB1_tuned = model_B1
# pred_B1 = classifierB1_tuned.predict(te_X_B1)
# acc_B1_test = accuracy_score(te_Y_B1, pred_B1)



# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
# from B2 import B2_model
# model_B2 = B2_model.SVM_B2(tr_X_B2, tr_Y_B2)
# acc_B2_train, classifierB2_tuned = model_B2
# pred_B2 = classifierB2_tuned.predict(te_X_B2)
# acc_B2_test = accuracy_score(te_Y_B2, pred_B2)



# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
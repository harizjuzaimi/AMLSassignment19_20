import os

# ======================================================================================================================
# Data preprocessing

from data_preprocessing import get_data_A1, get_data_A2, get_data_B1, get_data_B2
tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1 = get_data_A1()


tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = get_data_A2()


tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1 = get_data_B1()


tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = get_data_B2()

# ======================================================================================================================
# Task A1
from A1 import A1_model
model_A1 = A1_model.SVM_A1(tr_X_A1, tr_Y_A1, te_X_A1, te_Y_A1)
acc_A1_train, acc_A1_test = model_A1

# model_A1 = A1(args...)                 # Build model object.
# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
from A2 import A2_model
model_A2 = A2_model.SVM_A2(tr_X_A2, tr_Y_A2, te_X_A2, te_Y_A2)
acc_A2_train, acc_A2_test = model_A2

# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
from B1 import B1_model
model_B1 = B1_model.SVM_B1(tr_X_B1, tr_Y_B1, te_X_B1, te_Y_B1)
acc_B1_train, acc_B1_test = model_B1

# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
from B2 import B2_model
model_B2 = B2_model.SVM_B2(tr_X_B2, tr_Y_B2, te_X_B2, te_Y_B2)
acc_B2_train, acc_B2_test = model_B2

# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'
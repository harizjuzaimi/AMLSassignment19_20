import time
start_time = time.time()
# This whole process will took 20.6 minutes :)
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
acc_A1_train, classifierA1_tuned = A1_model.train(tr_X_A1, tr_Y_A1)
acc_A1_test = A1_model.test(classifierA1_tuned, te_X_A1, te_Y_A1)


from A1 import A1_add
temp_X_A1, temp_Y_A1 = A1_add.get_data()
acc_A1_add_test = A1_model.test(classifierA1_tuned, temp_X_A1, temp_Y_A1)

# ======================================================================================================================
# Task A2
from A2 import A2_model
acc_A2_train, classifierA2_tuned = A2_model.train(tr_X_A2, tr_Y_A2)
acc_A2_test = A2_model.test(classifierA2_tuned, te_X_A2, te_Y_A2)

from A2 import A2_add
temp_X_A2, temp_Y_A2 = A2_add.get_data()
acc_A2_add_test = A2_model.test(classifierA2_tuned, temp_X_A2, temp_Y_A2)

# ======================================================================================================================
# Task B1
from B1 import B1_model
acc_B1_train, classifierB1_tuned = B1_model.train(tr_X_B1, tr_Y_B1)
acc_B1_test = B1_model.test(classifierB1_tuned, te_X_B1, te_Y_B1)

from B1 import B1_add
temp_X_B1, temp_Y_B1 = B1_add.get_data()
acc_B1_add_test = B1_model.test(classifierB1_tuned, temp_X_B1, temp_Y_B1)

# ======================================================================================================================
# Task B2
from B2 import B2_model
acc_B2_train, classifierB2_tuned = B2_model.train(tr_X_B2, tr_Y_B2)
acc_B2_test = B2_model.test(classifierB2_tuned, te_X_B2, te_Y_B2)

from B2 import B2_add
temp_X_B2, temp_Y_B2 = B2_add.get_data()
acc_B2_add_test = B2_model.test(classifierB2_tuned, temp_X_B2, temp_Y_B2)

stop_time = time.time()
print(stop_time - start_time)

# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

print('Additional Test for - A1: {}; A2: {}; B1: {}; B2: {}'.format(acc_A1_add_test, acc_A2_add_test, acc_B1_add_test, acc_B2_add_test))

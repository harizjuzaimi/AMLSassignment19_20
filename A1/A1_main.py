import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

os.chdir('..')
print(os.path.abspath(os.curdir))
from data_A1 import get_data
tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1 = get_data()


from A1 import SVM_A1
model_A1 = SVM_A1(tr_X_A1, tr_Y_A1, te_X_A1, te_Y_A1)

# Clean up memory/GPU etc...             # Some code to free memory if necessary.


acc_A1_train, acc_A1_test = model_A1

print('TA1:{},{}'.format(acc_A1_train, acc_A1_test))




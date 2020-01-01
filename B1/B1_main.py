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
from data_A2 import get_data
tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1 = get_data()


from B1_model import SVM_B1
model_B1 = SVM_B1(tr_X_B1, tr_Y_B1, te_X_B1, te_Y_B1)

# Clean up memory/GPU etc...             # Some code to free memory if necessary.


acc_B1_train, acc_B1_test = model_B1

print('TB1:{},{}'.format(acc_B1_train, acc_B1_test))




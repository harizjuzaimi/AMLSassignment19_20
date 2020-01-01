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
tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = get_data()


from A2_model import SVM_A2
model_A2 = SVM_A2(tr_X_A2, tr_Y_A2, te_X_A2, te_Y_A2)

# Clean up memory/GPU etc...             # Some code to free memory if necessary.


acc_A2_train, acc_A2_test = model_A2

print('TA2:{},{}'.format(acc_A2_train, acc_A2_test))




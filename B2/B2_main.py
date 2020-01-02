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
from data_B2 import get_data
tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = get_data()


from B2_model import SVM_B2
model_B2 = SVM_B2(tr_X_B2, tr_Y_B2, te_X_B2, te_Y_B2)

# Clean up memory/GPU etc...             # Some code to free memory if necessary.


acc_B2_train, acc_B2_test = model_B2

print('TB2:{},{}'.format(acc_B2_train, acc_B2_test))




import pickle
from sklearn.model_selection import train_test_split
import os

os.chdir("./Data")
def get_data():

    with open('X_A1.dat', 'rb') as f:
        X_A1 = pickle.load(f)

    with open('Y_A1.dat', 'rb') as f:
        Y_A1 = pickle.load(f)

    tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1 = train_test_split(X_A1, Y_A1, test_size=0.3)

    return tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1
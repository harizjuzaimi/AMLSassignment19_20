import pickle
from sklearn.model_selection import train_test_split
import os

os.chdir("./Data")

def get_data():

    with open('X_B1.dat', 'rb') as f:
        X_B1_load = pickle.load(f)

    with open('Y_B1.dat', 'rb') as f:
        Y_B1_load = pickle.load(f)

    tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1 = train_test_split(X_B1_load, Y_B1_load, test_size=0.3)

    return tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1
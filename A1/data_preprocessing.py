import pickle
from sklearn.model_selection import train_test_split
import os

os.chdir("./Data")
def get_data():

    with open('X_saved.dat', 'rb') as f:
        X_load = pickle.load(f)

    with open('Y_saved.dat', 'rb') as f:
        Y_load = pickle.load(f)

    tr_X, te_X, tr_Y, te_Y = train_test_split(X_load, Y_load, test_size=0.3)

    return tr_X, te_X, tr_Y, te_Y
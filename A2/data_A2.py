import pickle
from sklearn.model_selection import train_test_split
import os

os.chdir("./Data")
def get_data():

    with open('X_A2.dat', 'rb') as f:
        X_A2 = pickle.load(f)

    with open('Y_A2.dat', 'rb') as f:
        Y_A2 = pickle.load(f)

    tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = train_test_split(X_A2, Y_A2, test_size=0.3)

    return tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2
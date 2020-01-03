import pickle
from sklearn.model_selection import train_test_split
import os


os.chdir('./Data')

def get_data_A1():

    with open('X_A1.dat', 'rb') as f:
        X_A1 = pickle.load(f)

    with open('Y_A1.dat', 'rb') as f:
        Y_A1 = pickle.load(f)

    tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1 = train_test_split(X_A1, Y_A1, test_size=0.3)

    return tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1

def get_data_A2():

    with open('X_A2.dat', 'rb') as f:
        X_A2 = pickle.load(f)

    with open('Y_A2.dat', 'rb') as f:
        Y_A2 = pickle.load(f)

    tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = train_test_split(X_A2, Y_A2, test_size=0.3)

    return tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2

def get_data_B1():

    with open('X_B1.dat', 'rb') as f:
        X_B1 = pickle.load(f)

    with open('Y_B1.dat', 'rb') as f:
        Y_B1 = pickle.load(f)

    tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1 = train_test_split(X_B1, Y_B1, test_size=0.3)

    return tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1

def get_data_B2():

    with open('X_B2.dat', 'rb') as f:
        X_B2= pickle.load(f)

    with open('Y_B2.dat', 'rb') as f:
        Y_B2 = pickle.load(f)

    tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = train_test_split(X_B2, Y_B2, test_size=0.3)

    return tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2

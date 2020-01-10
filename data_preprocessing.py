import pickle
from sklearn.model_selection import train_test_split
import os

basedir = os.path.abspath(os.curdir)
data_dir = os.path.join(basedir, 'Data')

def get_data_A1():

    X_A1_dir = os.path.join(data_dir, 'X_A1.dat')
    Y_A1_dir = os.path.join(data_dir, 'Y_A1.dat')

    with open(X_A1_dir, 'rb') as f:
        X_A1 = pickle.load(f)

    with open(Y_A1_dir, 'rb') as f:
        Y_A1 = pickle.load(f)

    tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1 = train_test_split(X_A1, Y_A1, test_size=0.3)

    return tr_X_A1, te_X_A1, tr_Y_A1, te_Y_A1

def get_data_A2():

    X_A2_dir = os.path.join(data_dir, 'X_A2.dat')
    Y_A2_dir = os.path.join(data_dir, 'Y_A2.dat')

    with open(X_A2_dir, 'rb') as f:
        X_A2 = pickle.load(f)

    with open(Y_A2_dir, 'rb') as f:
        Y_A2 = pickle.load(f)

    tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2 = train_test_split(X_A2, Y_A2, test_size=0.3)

    return tr_X_A2, te_X_A2, tr_Y_A2, te_Y_A2

def get_data_B1():

    X_B1_dir = os.path.join(data_dir, 'X_B1.dat')
    Y_B1_dir = os.path.join(data_dir, 'Y_B1.dat')

    with open(X_B1_dir, 'rb') as f:
        X_B1 = pickle.load(f)

    with open(Y_B1_dir, 'rb') as f:
        Y_B1 = pickle.load(f)

    tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1 = train_test_split(X_B1, Y_B1, test_size=0.3)

    return tr_X_B1, te_X_B1, tr_Y_B1, te_Y_B1

def get_data_B2():

    X_B2_dir = os.path.join(data_dir, 'X_B2.dat')
    Y_B2_dir = os.path.join(data_dir, 'Y_B2.dat')

    with open(X_B2_dir, 'rb') as f:
        X_B2= pickle.load(f)

    with open(Y_B2_dir, 'rb') as f:
        Y_B2 = pickle.load(f)

    tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = train_test_split(X_B2, Y_B2, test_size=0.3)

    return tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2


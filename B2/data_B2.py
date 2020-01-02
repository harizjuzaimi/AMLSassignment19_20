import pickle
from sklearn.model_selection import train_test_split
import os

os.chdir("./Data")
def get_data():

    with open('X_B2.dat', 'rb') as f:
        X_B2= pickle.load(f)

    with open('Y_B2.dat', 'rb') as f:
        Y_B2 = pickle.load(f)

    tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2 = train_test_split(X_B2, Y_B2, test_size=0.3)

    return tr_X_B2, te_X_B2, tr_Y_B2, te_Y_B2
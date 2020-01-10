import numpy as np
from sklearn.preprocessing import StandardScaler
from B1 import B1_landmark

def get_data():

    X_B1, y_B1 = B1_landmark.extract_features_labels()
    Y_B1 = np.array([y_B1, -(y_B1 - 1)]).T
    temp = X_B1.reshape(len(X_B1), 17*2)
    scaler = StandardScaler()
    temp_X_B1 = scaler.fit_transform(temp)
    temp_Y_B1 = list(zip(*Y_B1))[0]

    return temp_X_B1, temp_Y_B1
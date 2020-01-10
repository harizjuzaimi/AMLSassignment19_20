import numpy as np
from sklearn.preprocessing import StandardScaler
from A1 import A1_landmarks

def get_data():

    X_A1, y_A1 = A1_landmarks.extract_features_labels()
    Y_A1 = np.array([y_A1, -(y_A1 - 1)]).T
    temp = X_A1.reshape(len(X_A1), 68*2)
    scaler = StandardScaler()
    temp_X_A1 = scaler.fit_transform(temp)
    temp_Y_A1 = list(zip(*Y_A1))[0]

    return temp_X_A1, temp_Y_A1
import numpy as np
from sklearn.preprocessing import StandardScaler
from A2 import A2_landmarks

def get_data():

    X_A2, y_A2 = A2_landmarks.extract_features_labels()
    Y_A2 = np.array([y_A2, -(y_A2 - 1)]).T
    temp = X_A2.reshape(len(X_A2), 68*2)
    scaler = StandardScaler()
    temp_X_A2 = scaler.fit_transform(temp)
    temp_Y_A2 = list(zip(*Y_A2))[0]

    return temp_X_A2, temp_Y_A2
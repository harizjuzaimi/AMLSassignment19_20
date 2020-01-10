import numpy as np
from sklearn.preprocessing import StandardScaler
from B2 import B2_landmarks

def get_data():

    X_B2, temp_Y_B2 = B2_landmarks.extract_features_labels()
    scaler = StandardScaler()
    temp_X_B2 = scaler.fit_transform(X_B2)

    return temp_X_B2, temp_Y_B2
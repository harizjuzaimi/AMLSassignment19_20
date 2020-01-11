# UCL Applied Machine Learning Systems Assignment
Student Number: 160072927

# Assignment Task
- Task A1 <br />
Gender detection (Male/Female)
- Task A2 <br />
Emotion detection (Smiling/Not Smiling)
- Task B1 <br />
Face shape recognition (5 types of face shapes)
- Task B2 <br />
Eye color recognition (5 types of color)

# Files Description
**Python Files**
- model.py<br />
Main model for each task.s SVM classifier. The file is used for implementation of training and testing
- landmarks.py<br />
Implements feature extraction
- add.py<br />
To test on additional dataset. Pre processing file for new dataset<br />

**Jupyter Files**
- test.ipynb<br />
The purpose of this file is during designing and building model phase. This is also for debugging code
- data.ipynb<br />
Data preprocessing. Training data will be upload, feature extracted, scale and saved
- picture.ipynb<br />
Illustrates example of images for feature extraction<br />

**Folder**
- Data
  - Contains additional dataset that were sent on 9th January<br />
- train_shape
  - contains train_eyes.py and train_chin.py that were used to train custom shape predictor

**Landmarks .dat file**<br />
chin.dat, eye.dat and shape_predictor_68_face_landmarks.dat is the file use for shape predictor for task B1, B2, and both A1, A2 respectively

# Required Libraries
Scikit-learn, Scikit-image, opencv-python, Dlib, Numpy, Keras, pickle, os

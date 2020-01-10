import os
import skimage
from skimage import draw
import cv2
import dlib
import numpy as np

# PATH TO ALL IMAGES
global basedir, temp_dir, image_paths, target_size
basedir = './Data'
temp_dir = os.path.join(basedir, 'cartoon_set_test')
images_dir = os.path.join(temp_dir, 'img')
labels_filename = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('eyes.dat')


def extract_features_labels():
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open(os.path.join(temp_dir, labels_filename), 'r')
    lines = labels_file.readlines()
    colour_labels = {line.split('\t')[0]: int(line.split('\t')[1]) for line in lines[1:]}

    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        img_error = []
        for img_path in image_paths:
            bgr = []
            file_name = img_path.split('.')[1].split('\\')[-1]

            img = cv2.imread(img_path)
            res_img = img.astype('uint8')
            gray_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
            gray_img = gray_img.astype('uint8')

            rect = detector(gray_img, 1)
            num_check = len(rect)

            if num_check is not 0:
                rect = rect[0]
                sp = predictor(img, rect)
                landmarks = np.array([[p.x, p.y] for p in sp.parts()])

                # Select the landmarks that represents the shape of the face
                RIGHT_EYEBROW_POINTS = list(range(0, 6))
                outline = landmarks[RIGHT_EYEBROW_POINTS]

                # Draw a polygon using these landmarks using scikit-image
                Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])

                # Create a canvas with zeros and use the polygon as mask to original image
                mask = np.zeros(gray_img.shape, dtype=np.uint8)
                mask[Y, X] = 255

                colour = ('b', 'g', 'r')
                for i, col in enumerate(colour):
                    hist_mask = cv2.calcHist([img], [i], mask, [256], [0, 256])
                    bgr.append(np.argmax(hist_mask))

                all_features.append(bgr)
                all_labels.append(colour_labels[file_name])

            if num_check is 0:
                img_error.append(file_name)

    landmark_features = np.array(all_features)
    colour_labels = np.array(all_labels)  # simply converts the -1 into 0, so male=1 and female=0
    return landmark_features, colour_labels
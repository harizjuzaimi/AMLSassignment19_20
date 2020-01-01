import os
import skimage
from skimage import draw
from keras.preprocessing import image
import cv2
import dlib
import numpy as np

# PATH TO ALL IMAGES
global basedir, image_paths, target_size

os.chdir('..')
basedir = './Datasets'
images_dir = os.path.join(basedir, 'img_cartoon')
labels_filename = 'labels_cartoon.csv'

detector = dlib.get_frontal_face_detector()

temp_dir = './Data'
predictor_dir = os.path.join(temp_dir, 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def extract_features_labels():
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=1 and female=0) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()
    colour_labels = {line.split('\t')[0]: int(line.split('\t')[1]) for line in lines[1:]}
    # all_features = None
    # all_labels = None
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        gambar_error = []
        for img_path in image_paths:
            bgr = []
            file_name = img_path.split('.')[1].split('\\')[-1]

            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)

            if features is not None:
                img = cv2.imread(img_path)
                # Select the landmarks that represents the shape of the face
                RIGHT_EYEBROW_POINTS = list(range(36, 42))
                outline = landmarks[RIGHT_EYEBROW_POINTS]

                # Draw a polygon using these landmarks using scikit-image
                Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])

                # Create a canvas with zeros and use the polygon as mask to original image
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask[Y, X] = 255
                # masked_img = cv2.bitwise_and(img,img,mask = mask)

                colour = ('b', 'g', 'r')
                for i, col in enumerate(colour):
                    #
                    hist_mask = cv2.calcHist([img], [i], mask, [256], [0, 256])
                    bgr.append(np.argmax(hist_mask))

                all_features.append(bgr)
                all_labels.append(colour_labels[file_name])

            if features is None:
                gambar_error.append(file_name)

    landmark_features = np.array(all_features)
    colour_labels = np.array(all_labels)  # simply converts the -1 into 0, so male=1 and female=0
    return landmark_features, colour_labels, gambar_error



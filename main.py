import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import metrics
from PIL import Image
from PIL import ImageOps
import cv2
import os
import shutil
import math
import h5py

from keras import backend

import glob


# parse image by image
# apply bounding boxes to all the images to find all the street signs
# localize street signs, save them and then pass them into the CNN to be learned/classified

# grayImg = imgaussfilt(grayImg,1.6);
# BWImg = edge(grayImg); contour detection
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

# next thing we need to do is find a suitable way to localize the traffic signs such as road signs or even
# traffic lights with high accuracy two ways of doing it, firstly, we can do it through a heuristic approach
# where we define some weights that are telling about the sign itself, squareness, where it is located in the
# picture as in the more right it is on the screen the more likely it is to be a road sign
# or we can take a regression approach which I will look into.
def nothing(x):
    pass


def finding_general_canny_thresholds(filename):
    rgb_image = np.array(Image.open(filename))  # rgb image to be greyscaled
    grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grey_image, (3, 3), 2)

    img = cv2.imread(filename)
    cv2.namedWindow('canny')
    switch1 = 'off 1'
    switch2 = 'off 2'

    cv2.createTrackbar(switch1, 'canny', 0, 1, nothing)
    cv2.createTrackbar('lower1', 'canny', 0, 255, nothing)
    cv2.createTrackbar('upper1', 'canny', 0, 255, nothing)

    cv2.createTrackbar(switch2, 'canny', 0, 1, nothing)
    cv2.createTrackbar('lower2', 'canny', 0, 255, nothing)
    cv2.createTrackbar('upper2', 'canny', 0, 255, nothing)
    while (1):
        lower_thresh = cv2.getTrackbarPos('lower1', 'canny')
        upper_thresh = cv2.getTrackbarPos('upper1', 'canny')
        s_thresh = cv2.getTrackbarPos(switch1, 'canny')
        lower_canny = cv2.getTrackbarPos('lower2', 'canny')
        upper_canny = cv2.getTrackbarPos('upper2', 'canny')
        s_can = cv2.getTrackbarPos(switch2, 'canny')

        if s_can == 0 and s_thresh == 0:
            edges = blurred_image
        elif s_can == 1 and s_thresh == 0:
            edges = cv2.Canny(blurred_image, lower_canny, upper_canny)
        elif s_can == 0 and s_thresh == 1:
            edges = cv2.threshold(blurred_image, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]
        elif s_can == 1 and s_thresh == 1:
            thresh = cv2.threshold(blurred_image, lower_thresh, upper_thresh, cv2.THRESH_BINARY)[1]
            edges = cv2.Canny(thresh, lower_canny, upper_canny)
        cv2.imshow('canny', edges)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break


# finding_general_canny_thresholds('data/train_images/road797.png')

def get_training_images(path_to_csv):
    # we will use the csv file to localize the training images to be of just the sign
    # size of the training images needs to be 50 by 50
    destination_dir = "data/LeNet Training Images/"
    source_directory = 'data/images_resized'
    file_obj = open(path_to_csv)
    # the layout of the file should be: [id, pathname to image, class, x1, y1, x2, y2]
    training_data = []
    target_values = []  # one hot encoding of the class
    if os.path.isdir(destination_dir) is False:
        os.mkdir(destination_dir)
    i = 0
    for line in file_obj:
        if not i == 0:
            line = line.strip('\n').split(',')
            image = np.array(Image.open('data/' + line[1]))
            cropped_image_array = image[int(line[3]):int(line[5]), int(line[4]):int(line[6])]
            cropped_image = Image.fromarray(np.uint8(cropped_image_array))
            resized_cropped_image = cropped_image.resize((100, 100))  # resize the image
            resized_cropped_image_grey = ImageOps.grayscale(resized_cropped_image)  # grey scale the image
            test_img = np.array(resized_cropped_image_grey)
            training_data.append(test_img)  # here is the gray scaled image
            target = np.array(int(line[2]))
            target_values.append(target)  # convert to categorical
            image_path = (line[1].split('/'))[1]  # location of file
            resized_cropped_image_grey.save(destination_dir + image_path)  # save the image
        i += 1
    file_obj.close()
    return np.array(training_data[:-50]), np.array(target_values[:-50]), np.array(training_data[-50:]), np.array(
        target_values[-50:])  # training and validation data


#     # this dataset is an imbalanced dataset with 76% of the distribution belonging to class 1
#     # we will utilize resampling to fix this distribution issue
def oversample_data(training_data, train_target):
    i = 0
    class_appearances = [0, 0, 0, 0]
    while i < training_data.shape[0]:
        class_appearances[train_target[i]] += 1
        i += 1

    over_represented_class = class_appearances.index(max(class_appearances))  # over represented class
    print(class_appearances)
    new_training_data = []
    new_target_data = []
    class_rep_relative_to_class_0 = [class_appearances[0], int(class_appearances[0] / class_appearances[1]),
                                     int(class_appearances[0] / class_appearances[2]),
                                     int(class_appearances[0] / class_appearances[3])]
    i = 0
    while i < train_target.shape[0]:
        if train_target[i] != over_represented_class:
            # duplicate point
            x = 0
            # append duplicate points
            while x < class_rep_relative_to_class_0[train_target[i]]:
                # add the numerous more representations needed
                new_training_data.append(training_data[i])
                new_target_data.append(train_target[i])
                x += 1
        else:
            # just append the data point once if it is the overrepresented class
            new_training_data.append(training_data[i])
            new_target_data.append(train_target[i])
        i += 1

    return np.array(new_training_data), np.array(new_target_data)


def undersample(training_data, train_target):
    training_data, train_target, val_data, val_target = get_training_images('data/train.csv')
    i = 0
    class_appearances = [0, 0, 0, 0]
    while i < training_data.shape[0]:
        class_appearances[train_target[i]] += 1
        i += 1

    max_appearances = max(class_appearances)
    over_represented_class = class_appearances.index(max_appearances)  # over represented class
    desired_appearances = [int((class_appearances[1] + class_appearances[2] + class_appearances[
        3]) / 3)] + class_appearances[1:]
    i = 0
    counter = 0
    balanced_data_set = []
    balanced_target_set = []
    while i < train_target.shape[0]:
        # if the over represented class is indexed, then everytime it gets added to the dataset increment
        if train_target[i] == 0 and counter < desired_appearances[over_represented_class]:
            balanced_data_set.append(training_data[i])
            balanced_target_set.append(train_target[i])
            counter += 1
        elif train_target[i] != 0:
            balanced_data_set.append(training_data[i])
            balanced_target_set.append(train_target[i])
        i += 1
    return np.array(balanced_data_set), np.array(balanced_target_set)


def create_road_model():
    cnn_road = keras.Sequential(
        [
            layers.Conv2D(16, (12, 12), input_shape=(100, 100, 1), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
            layers.MaxPooling2D((4, 4)),
            layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(400, activation='relu', kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.L1(0.01), bias_regularizer=tf.keras.regularizers.L2(0.01)),
            layers.Dense(4, activation='softmax', activity_regularizer=tf.keras.regularizers.L1(0.01))  # there needs to be 4 outputs that indicate what type of sign we
        ]
    )
    return cnn_road


def cnn_road_model():
    # we will under sample the data to get a better representation of the classes in the training dataset
    training_data, train_target, val_data, val_target = get_training_images('data/train.csv')
    undersampled_training, train_target = oversample_data(training_data, train_target) # undersample training data
    undersampled_training = undersampled_training.reshape(undersampled_training.shape[0], 100, 100, 1)
    # over_sampled_training, oversampled_target = oversample_data(training_data, train_target)
    # over_sampled_training = over_sampled_training.reshape(over_sampled_training.shape[0], 100, 100, 1)
    # train_target = tf.keras.utils.to_categorical(oversampled_target)
    # training_data = training_data.reshape(training_data.shape[0], 100, 100, 1)
    train_target = tf.keras.utils.to_categorical(train_target)
    val_data_oversampled, val_target_oversampled = oversample_data(val_data, val_target) # oversample validation data
    val_data_oversampled = val_data_oversampled.reshape(val_data_oversampled.shape[0], 100, 100, 1)
    val_target_oversampled = tf.keras.utils.to_categorical(val_target_oversampled)
    val_data = val_data.reshape(val_data.shape[0], 100, 100, 1)
    val_target = tf.keras.utils.to_categorical(val_target)

    print("The amount of elements in x and the target should be exactly the same")
    print("Training Data:")
    print(undersampled_training.shape)
    print(train_target)
    print("Training Data:")
    print(len(val_data_oversampled))
    print(len(val_target_oversampled))

    # Create a callback that saves the model's weights
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='my_model_weights.h5',
        save_weights_only=True,
        monitor='val_precision',
        mode='max',
        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_precision',
        verbose=1,
        patience=15,
        mode='max',
        restore_best_weights=True)

    # convolutional model
    cnn_road = create_road_model()
    cnn_road.summary()

    cnn_road.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[metrics.Precision(name='precision'), metrics.BinaryAccuracy(name='accuracy')]
    )

    history = cnn_road.fit(
        undersampled_training,
        train_target,
        batch_size=20,
        epochs=100,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(val_data_oversampled, val_target_oversampled),
        callbacks=[early_stopping, model_checkpoint_callback]
    )
    # get training images and validation images


#cnn_road_model()

def localization_and_classification():
    cnn_road = create_road_model()
    cnn_road.load_weights('my_model_weights.h5')  # for the best weights
    localized_signs = heuristic_localization(
        'data/test_images/road193.png')  # in the form of imgs must convert to np array
    pred_list = []
    if localized_signs != 0:
        for point in localized_signs:
            pred_list.append(point[2].reshape(100, 100, 1))
    pred_list = np.array(pred_list)
    predictions = cnn_road.predict(np.array(pred_list)) # make predictions
    # do nothing if there are no signs
    if localized_signs != 0:
        pred_indexer = 0
        # do some processing so we can achieve the bounding boxes
        for point in localized_signs:
            coord_1 = point[0]
            coord_2 = point[1]
            if max(predictions[pred_indexer]) >= .5:
                # is not noise
                label = prediction_to_string(predictions[pred_indexer])
                bounding_box_given_with_label(point[3], coord_1, coord_2, label)
            else:
                # noise and do not put bounding box
                print('no sign detected')

            pred_indexer += 1


def prediction_to_string(prediction):
    # [1, 0, 0, 0]
    if math.ceil(prediction[0]) == 1:
        label = "speed sign"
    elif math.ceil(prediction[1]) == 1:
        label = 'stop'
    elif math.ceil(prediction[2]) == 1:
        label = 'crosswalk'
    else:
        label = 'traffic light'
    return label


def heuristic_localization(file_path):
    # image processing
    rgb_image = np.array(Image.open(file_path))  # rgb image to be greyscaled
    grey_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)  # greyscale
    blurred_image = cv2.GaussianBlur(grey_image, (5, 5), 1)  # slightly blur image
    img_canny = cv2.Canny(blurred_image, threshold1=125,
                          threshold2=180)  # does not matter what the thresholds are as the image has been binarized
    cv2.imshow("test", img_canny)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    circle_list = []
    square_list = []
    # localize the speed signs or stop signs
    num_of_contours = len(contours)  # debugging purposes
    localized_cropped_images = []
    z = 0
    while z <= 1:
        for contour in contours:
            if z == 0:
                approx = cv2.approxPolyDP(contour, .01 * cv2.arcLength(contour, True), True)
            else:
                approx = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            # if something seems like a circle append it
            # chosen through trial and error and comparing to the extreme cases
            i = 0

            # squares
            if len(approx) == 4 and area > 100:
                square_list.append(contour)
                x, y, w, h = cv2.boundingRect(contour)
                x_1, y_1, x_2, y_2 = int(x - w / 5), int(y - h / 5), int(x + w + w / 5), int(y + h + h / 5)
                im = cv2.resize(grey_image[y:y + w, x:x + w], (100, 100))
                cv2.imshow('test', im)
                cv2.waitKey(0)
                localized_cropped_images.append([(x_1, y_1), (x_2, y_2), im, rgb_image])
                cv2.rectangle(rgb_image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
            # approach for circles
            if len(approx) > 11 and len(approx) < 14 and area > 250:
                circle_list.append(contour)
                x, y, w, h = cv2.boundingRect(contour)
                x_1, y_1, x_2, y_2 = int(x - w / 5), int(y - h / 5), int(x + w + w / 5), int(y + h + h / 5)
                im = cv2.resize(grey_image[y:y + w, x:x + w], (100, 100))
                cv2.imshow('test', im)
                cv2.waitKey(0)
                localized_cropped_images.append([(x_1, y_1), (x_2, y_2), im, rgb_image])
                cv2.rectangle(rgb_image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
        cv2.drawContours(rgb_image, square_list, -1, (255, 0, 0), 2)
        cv2.drawContours(rgb_image, circle_list, -1, (255, 0, 0), 2)
        z += 1
    cv2.imshow('Objects Detected', rgb_image)
    cv2.waitKey(0)
    return localized_cropped_images

#heuristic_localization('data/test_images/road193.png')
#heuristic_localization('data/train_images/road398.png')
def bounding_box_given_with_label(image, coordinate_tuple_1, coordinate_tuple_2, label):
    # given a bounding box predicted by the net, the box will be drawn on the corresponding,
    # images where they can be visually checked for accuracy
    # furthermore, the classification will also be shown somewhere on that exact same image
    color = (255, 0, 0)
    fixed_tuple_coord_1 = (coordinate_tuple_1[1], coordinate_tuple_1[0])  # have to flip the coordinates because of
    # how opencv justifies their image
    fixed_tuple_coord_2 = (coordinate_tuple_2[1], coordinate_tuple_2[0])  # same here
    # draw bounding box
    boxed_image = cv2.rectangle(image, fixed_tuple_coord_1, fixed_tuple_coord_2, color=color, thickness=1)

    cv2.putText(boxed_image, text=label, org=coordinate_tuple_2,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.4, color=(255, 0, 0),
                thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('Prediction', boxed_image)  # show the image with the bounding box over the sign
    cv2.waitKey(0)  # wait for key press to be dismissed


def bounding_box(image_path_name, coordinate_tuple_1, coordinate_tuple_2):
    # given a bounding box predicted by the net, the box will be drawn on the corresponding,
    # images where they can be visually checked for accuracy
    # furthermore, the classification will also be shown somewhere on that exact same image
    color = (255, 0, 0)
    image = np.array(Image.open(image_path_name))
    fixed_tuple_coord_1 = (coordinate_tuple_1[1], coordinate_tuple_1[0])  # have to flip the coordinates because of
    # how opencv justifies their image
    fixed_tuple_coord_2 = (coordinate_tuple_2[1], coordinate_tuple_2[0])  # same here
    # draw bounding box
    boxed_image = cv2.rectangle(image, fixed_tuple_coord_1, fixed_tuple_coord_2, color=color, thickness=1)
    cv2.imshow(image_path_name, boxed_image)  # show the image with the bounding box over the sign
    cv2.waitKey(0)  # wait for key press to be dismissed

# bounding_box('data/train_images/road875.png', (186,74), (195,91))

def seperate_data(file_name):
    destination_dir = "data/train_images/"
    if os.path.isdir(destination_dir) is False:
        os.mkdir(destination_dir)

    file_obj = open(file_name)
    i = 0
    print(os.getcwd())
    for line in file_obj:
        if i == 0:
            i += 1
        else:
            line = line.strip('\n').split(',')
            dir_and_file_name = line[1].split('/')  # [dir, file_name]
            shutil.copyfile('data/' + line[1], destination_dir + '/' + dir_and_file_name[1])


def upscale_images():
    file_path = 'data/obj/'
    for file_name in os.listdir(file_path):
        if file_name.endswith('.png'):
            with Image.open(file_path + file_name) as im:
                im.resize((448, 300)).save(file_path + file_name)  # width by height
                im.close()


def pre_process_data():
    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    print(current_dir)

    current_dir = 'data'

    # Percentage of images to be used for the test set
    percentage_test = 10;

    # Create and/or truncate train.txt and test.txt
    file_train = open('data/obj/train.txt', 'w')
    file_test = open('data/obj/test.txt', 'w')

    # Populate train.txt and test.txt
    counter = 1
    index_test = round(100 / percentage_test)
    dir = current_dir + "/train_images"

    for file_name in os.listdir(dir):

        if counter == index_test:
            counter = 1
            file_test.write("data/obj" + "/" + file_name + "\n")
        else:
            file_train.write("data/obj" + "/" + file_name + "\n")
            counter = counter + 1


# pre_process_data()

def more_data_processing():
    file_name = 'data/train.csv'
    file_obj = open(file_name)
    for line in file_obj:
        if 'new_path' not in line:
            line = line.strip('\n').split(',')
            file_name = line[1].strip('images_resized/').strip('/.png')
            # normalize the x and y coordinate
            if ('oad875' in file_name):
                print('here')
            # FIX THIS PLEASE
            # find the new x_1 and x_2 based off of the change in dimension of the image
            # new width = 320, old width is 300, new height is 448, old height is 447
            # coordinate_point_1 = (float(line[3])*(320/300), float(line[4])*(448/447))
            # coordinate_point_2 = (float(line[5])*(320/300), float(line[6])*(448/447))
            coordinate_point_1 = (float(line[3]), float(line[4]))
            coordinate_point_2 = (float(line[5]), float(line[6]))
            ##############
            width = abs(coordinate_point_1[0] - coordinate_point_2[0]) / 320
            height = abs(coordinate_point_1[1] - coordinate_point_2[1]) / 448
            x_center = (coordinate_point_1[0] + abs(coordinate_point_1[0] - coordinate_point_2[0]) / 2) / 320
            y_center = (coordinate_point_1[1] + abs(coordinate_point_1[1] - coordinate_point_2[
                1]) / 2) / 448  # normalized over the width and height of the image

            parameters = ' '.join([line[2], str(x_center), str(y_center), str(width), str(height)])

            file_write = open('data/obj/r' + file_name + '.txt', 'w')
            file_write.write(parameters)
            file_write.close()


# more_data_processing()

def check_bounding_box(image_path, corresponding_target_path):
    target_file = open('data/obj/' + corresponding_target_path)
    image_file = Image.open('data/obj/' + image_path)
    image_matrix = np.array(image_file)
    print(image_matrix.shape)
    target_values = target_file.readline().strip('\n').split(' ')  # [class, xcenter, ycenter, width, height]
    coordinate_top_left = (int((float(target_values[1]) * 320 - float(target_values[3]) * 320 / 2)),
                           int((float(target_values[2]) - float(target_values[4]) / 2) * 448))  # x_1, y_1
    coordinate_bot_right = (int((float(target_values[1]) + float(target_values[3]) / 2) * 320),
                            int((float(target_values[2]) + float(target_values[4]) / 2) * 448))
    bounding_box('data/obj/' + image_path, coordinate_top_left, coordinate_bot_right)


# check_bounding_box('road624.png', 'road624.txt')

# more_data_processing()
# upscale_images()
# test()
# bounding_box(image_path_name="data/images_resized/road44.png", coordinate_tuple_1=(37, 158),
#             coordinate_tuple_2=(71, 192))
# mnist_model()
# main()

localization_and_classification()

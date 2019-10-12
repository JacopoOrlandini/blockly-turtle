import os
import sys

import cv2
import numpy as np
import tensorflow as tf


# class definition

class Marker:
    def __init__(self, marker_id, corners, key, value):
        self.id = marker_id
        self.value = value
        self.corners = corners
        self.key = key


#################################

def take_key(marker):
    return marker.key


def sort_markers_vertical(markers):
    return sorted(markers, key=take_key)


def print_markers(markers):
    for i in range(len(markers)):
        print("Marker ID: ", markers[i].id, "Key: ", markers[i].key)


# read and resize image
def resize_image(img):
    height, width, depth = img.shape
    W = 1200.
    H = 1200.
    imgScalew = W / width
    imgScaleh = H / width
    newX, newY = img.shape[1] * imgScalew, img.shape[0] * imgScaleh
    return cv2.resize(img, (int(newX), int(newY)))


def fill_instructions_list(markers):
    instructions = []
    for i in range(len(markers)):
        instructions.append({markers[i].id: markers[i].value})
    return instructions


def RGB2HEX(color):
    """
    Conversion from RGB to Hex
    """
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_colors(img):
    """
    Method to retrieve the colors next to a marker.
    The basic method proposed uses KMeans clusterization.
    @param img : image imported from the main method.
    In number of colors we have a N dimensional space to search the Kmeans center.
    The return of this method is a Color (string)
    """
    from collections import Counter
    from sklearn.cluster import KMeans

    number_of_colors = 3
    modified_image = cv2.resize(img, (30, 30), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)
    clf = KMeans(n_clusters=number_of_colors)
    labels = clf.fit_predict(modified_image)
    cv2.imshow("g", img)
    cv2.waitKey(0)
    counts = Counter(labels)
    """
    Counter to count the key inside the cluster centers
    """
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    bgr = [ordered_colors[i] for i in counts.keys()]

    print(ordered_colors)
    print(bgr[:2])  # to exclude the white background we skip last element of 3 clusters (num_size_cluster)

    # BGR color
    for bgr_center in bgr[:2]:
        """
        Inside you can mesh the color to obtain primary or secondary colors.
        Or what you want to retrieve.
        """
        if abs(bgr_center[0] - bgr_center[1]) < 50 and abs(bgr_center[1] - bgr_center[2]) < 50:
            """
            This is the gray color. To be reviewed.
            """
            pass
        elif bgr_center[0] > 160:
            return "blue"
        elif bgr_center[1] > 160:
            return "green"
        elif bgr_center[2] > 160:
            return "red"
    # if no color was found return None to predict digits.
    return None


def predict_values(mask_img, vis_mask=False, visualise=False):
    """
    Prediction of multi digits inside a ROI (mask_img).
    Frozen model is loaded in ./module
    For optimization if a color is detected, the predicting values is bypassed.
    else we run the full method for the ROI next to the marker and we predict the value
    in the drawing box (rects).
    return pred if a multi digits is detected.
    else none
    """
    from tensorflow.python.platform import gfile
    from skimage.feature import hog

    bgr_colors = get_colors(mask_img)
    if bgr_colors is not None:
        return bgr_colors

    sess = tf.InteractiveSession()

    f = gfile.FastGFile("../model/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()

    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)

    """To retrive correct Ops
    for op in sess.graph.get_operations():
        print(op.name, op.type)
    """
    softmax_tensor = sess.graph.get_tensor_by_name('import/dense_2_1/Softmax:0')

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    tensor = []

    rects = sorted(rects, key=lambda ax: ax[3] + ax[0], reverse=True)

    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(mask_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # Resize the image
        if roi.shape[0] >= 40 and roi.shape[1] >= 40:
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            tensor.append(roi)

    x_test = tensor
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(tensor), 28, 28, 1)
    x_test = x_test.astype('float32')
    x_test /= 255

    predictions = sess.run(softmax_tensor, {'import/conv2d_1_input_1:0': x_test})
    preds = [p.argmax() for p in predictions]
    preds.reverse()
    res = 0
    if len(preds) <= 0:
        res = None
    else:
        preds = [str(x) for x in preds]
        res = int("".join(preds))
    sess.close()
    return res


def maxing(corner):
    """
    Return the Limits for the size rigth of the marker. 
    To understand the idea of this you must debug the point returning from the detection marker.
    Else mantain this method 
    """
    x1 = corner[0][0]
    y1 = corner[0][1]
    x2 = corner[1][0]
    y2 = corner[1][1]
    x3 = corner[2][0]
    y3 = corner[2][1]
    x4 = corner[3][0]
    y4 = corner[3][1]

    min_x = min([x1, x2, x3, x4])
    max_x = max([x1, x2, x3, x4])
    max_y = max([y1, y2, y3, y4])
    min_y = min([y1, y2, y3, y4])

    return min_x, max_x, min_y, max_y


def offset_right(corner, resized_img):
    """
    Method to limit the error from the altitude when the photo is taken.
    Is a sort of adjusting height for the camera.
    If this method is not runned, if you take photo from different size the marker size is not a 
    good choice to detect the right part drawing box.
    For example the marker has 10x10 pixel, the drawing box is moved to the right by 10.
    If the marker has 50x50 pixel, the drawing box is moved to the right by 50.
    This method is an abstraction of this use case.
    
    return offset position for the ROI.
    """
    min_x, max_x, min_y, max_y = maxing(corner)

    scale_x = max_x - min_x
    scale_y = max_y - min_y

    coeff_x = int(resized_img.shape[0] / scale_x)

    x1 = int(max_x)
    x2 = int(max_x) + int(scale_x + coeff_x) * 3

    y1 = int(min_y) - int(scale_y - 5)
    y2 = int(min_y) + int(scale_y * 1.5)

    if y2 - y1 < 50:
        y1 = int(min_y - 60)
        y2 = int(min_y + 60)

    return x1, x2, y1, y2


def get_roi_numbers(corner, resized_img):
    """
    Method to get the ROI image 
    return the masked image
    """
    x1, x2, y1, y2 = offset_right(corner, resized_img)
    mask_img = resized_img[y1:y2, x1:x2]

    return mask_img


# ritorna la lista di ID: lista di istruzioni
def aruco_detection(path, visualise=False):
    img = cv2.imread(path)
    resized_img = resize_image(img)

    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    markers = []

    """
    This is the invocation from my code and return a list of tuple with the information for the drawing box.
    """
    for i in range(len(corners)):
        print("Processing corner: {}/{}".format(i + 1, len(corners) + 1))
        corners_of_one_marker = corners[i][0]

        # [Start Detection Numbers Aruco]
        mask_img = get_roi_numbers(corners_of_one_marker, resized_img)
        digits = predict_values(mask_img, vis_mask=True, visualise=False)
        # [End Dectection Numbers Aruco]
        key = corners[i][0][0][1]
        marker_id = ids[i][0]
        print(digits, marker_id)

        new_marker = Marker(marker_id, corners_of_one_marker, key, digits)
        markers.append(new_marker)

    sorted_markers = sort_markers_vertical(markers)
    instructions = fill_instructions_list(sorted_markers)

    return instructions


"""
Put some image to test. 
The image for testing are inside the images folder. Go there and check what is available.
I create the image from an Ipad with note (are not real image for real blockly wooden pieces).

Print x to see all the tuples from the method.
"""
# __main__ entry point to launch detection
# print(aruco_detection("images/btmaze10err.jpeg"))
x = aruco_detection("images/color.png", visualise=False)
print(x)

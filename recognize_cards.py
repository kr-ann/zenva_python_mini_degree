import cv2
import numpy as np
from filters_app_openCV import to_gray

""" First three functions are an example of basic things we use in the app. """


def bin_thresh(img, threshold):
    """ Applies binary thresholding to the picture + gaussian blurring for smoothing. """
    _, new_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)  # 255 is the max value

    # make some smoothing to get rid of unnecessary ~splashes of color
    # parameters are the width and height of gaussian kernel, and standard deviation in X and Y direction,
    # sigmaX & sigmaY respectively. If only  sigmaX is specified, sigmaY is taken as same as sigmaX.
    # If both are given as zeros, they are calculated from kernel size.
    new_img = cv2.GaussianBlur(new_img, (9, 9), 0)
    return new_img


def inv_bin_thresh(img, threshold):
    """ Inverted binary thresholding + gaussian blurring. """
    # the 1st returned val = our threshold; if we use cv.THRESH_OTSU, it's optimal threshold val found by the algorithm
    _, new_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)

    # make some smoothing to get rid of unnecessary ~splashes of color
    # parameters are the width and height of gaussian kernel, and standard deviation in X and Y direction,
    # sigmaX & sigmaY respectively. If only sigmaX is specified, sigmaY is taken as same as sigmaX.
    # If both are given as zeros, they are calculated from kernel size.
    new_img = cv2.GaussianBlur(new_img, (5, 5), 0)
    return new_img


def contour_detection(img, img_thresh):
    """ Finds contours and returns an initial picture (given as the first argument) with these contours. """
    # contours is a Python list of all the contours in the image
    modified_img, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # apply rectangular approximation to one of the contours (like a test)
    contour = contours[6]
    epsilon = 0.1 * cv2.arcLength(contour, True)  # bool value - whether the line is closed
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # parameters are: what contours to draw (-1 means all of them), colour, thickness
    return cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)


def reorder(pts):
    """ Reorder points in a rectangle in clockwise order to be consistent with OpenCV. """
    # pts is a numpy array tha looks like [ [[numX numY]] [[num num]] [[num num]] [[num num]] ]
    pts = pts.reshape((4, 2))  # make it look like [ [numX numY] [num num] [num num] [num num] ]
    pts_new = np.zeros((4, 2), np.float32)

    add = pts.sum(1)  # array like [ numX+numY num+num num+num num+num ]
    pts_new[0] = pts[np.argmin(add)]  # the dot that is the nearest to the (0, 0)
    pts_new[2] = pts[np.argmax(add)]  # the remotest one

    diff = np.diff(pts, 1)  # array like [ [numY-numX] [num-num] [num-num] [num-num] ]
    pts_new[1] = pts[np.argmin(diff)]
    pts_new[3] = pts[np.argmax(diff)]

    return pts_new


def preprocess(img):
    """ Image cleaning. """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # parameters: image, max val, adaptive method (mean/gaussian), threshold type (binary or inverted binary),
    # blockSize	(size of a pixel neighborhood that is used to calculate a threshold value for the pixel)
    # C - constant subtracted from the mean or weighted mean (used in an adaptive method)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 1)

    return thresh


def img_compare(A, B):
    """ Metric of image comparison. """
    A = cv2.GaussianBlur(A, (5, 5), 5)
    B = cv2.GaussianBlur(B, (5, 5), 5)
    diff = cv2.absdiff(A, B)  # absolute difference
    _, diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    return np.sum(diff)


def closest_card(model, img):
    """ Find the playing card that is the closest to img. """
    features = preprocess(img)
    closest_match = sorted(model.values(), key=lambda x: img_compare(x[1], features))[0]
    return closest_match[0]


def extract_cards(img, num_cards=4):
    cards = []
    gray = to_gray(img)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)  # looots of blur to detect only high-level contours
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort extracted contours by their size, pick the largest, a heuristic to get rid of small false (not card) contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_cards]

    # dimensions of a resulted rectangle
    new_dim = np.array([[0, 0], [449, 0], [449, 449], [0, 449]], np.float32)

    for card in contours:
        epsilon = 0.1 * cv2.arcLength(card, True)
        approx = cv2.approxPolyDP(card, epsilon, True)
        approx = reorder(approx)  # reorganized list of coordinates

        # calculates a 3x3 matrix of a perspective transform from four pairs of the corresponding points
        transform = cv2.getPerspectiveTransform(approx, new_dim)
        # applies a perspective transformation to an image (the last arg is the size of the output image)
        warp = cv2.warpPerspective(img, transform, (450, 450))

        cards.append(warp)

    return cards


def train(training_labels_filename='train.tsv', training_image_filename='train.png', num_training_cards=56):
    """ Collect training information for model. """
    model = {}

    labels = {}
    with open(training_labels_filename, 'r') as file:
        for line in file:
            key, num, suit = line.strip().split()
            labels[int(key)] = (num, suit)

    training_img = cv2.imread(training_image_filename)
    for i, card in enumerate(extract_cards(training_img, num_training_cards)):
        model[i] = (labels[i], preprocess(card))

    return model


if __name__ == '__main__':
    """ That's a previous example
    img = cv2.imread('example.jpg')  # read picture and convert it to gray scale
    img_gray = to_gray(img)

    img_thresh = bin_thresh(img_gray, 240)
    img_with_contours = contour_detection(img, img_thresh)
    cv2.imshow('contours', img_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    filename = 'test.jpg'
    num_cards = 4

    model = train()

    img = cv2.imread(filename)

    width = img.shape[0]
    height = img.shape[1]
    if width < height:
        # rotate the image by 90 degrees
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)

    # See images
    for i, c in enumerate(extract_cards(img, num_cards)):
        card = closest_card(model, c)
        cv2.imshow(str(card), c)
    cv2.waitKey(0)

    cards = [closest_card(model, c) for c in extract_cards(img, num_cards)]
    print(cards)
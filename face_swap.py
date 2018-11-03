import cv2
import sys
from filters_app_openCV import to_gray


def draw_found_objects(objects, img):
    """ Draws a rectangular contour of the given objects (faces/eyes) on the image, shows and returns it. """
    for object in objects:
        # 'object' is a list [x y width height], (x y) are coordinates of the top left corner of the rectangle with obj
        x, y, w, h = object
        # draws a rectangle given the args: img, top left corner, bottom right corner, color, thickness
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('objects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


def detect_faces(gray_img):
    """ Detects faces on a given gray image. """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # returns a class that we can use

    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 8)  # args: scale factor, min num of neighbours
    return faces


def detect_eyes(gray_face_img):
    """ The function detects eyes on the given (gray) picture of a face (and returns the list of coordinates). """
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray_face_img, 1.1, 3)
    return eyes


def two_face_swap(img, faces):
    """ Swaps two faces given the list of faces and an image and shows the result. """

    if len(faces) != 2:
        sys.exit('Please input an image with EXACTLY 2 faces!')

    x1, y1, w1, h1 = faces[0]
    x2, y2, w2, h2 = faces[1]

    face1 = img[y1:y1+h1, x1:x1+w1]  # crop the image to include only the given face (list slicing)
    face2 = img[y2:y2+h2, x2:x2+w2]

    # change shape of images to fit into each other's places
    face1 = cv2.resize(face1, (w2, h2))
    face2 = cv2.resize(face2, (w1, h1))

    # face 1
    img[y1:y1+h1, x1:x1+w1] = face2

    # face 2
    img[y2:y2+h2, x2:x2+w2] = face1

    cv2.imshow('swap', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    # here we find a face on a basic picture
    img = cv2.imread('face.jpg')
    # the second arg is output image size, if (0,0) - it's computed from img.size(), fx, and fy
    # fx, fy - scale factors along the X and Y axes respectively
    cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = to_gray(img)
    faces = detect_faces(gray)
    _ = draw_found_objects(faces, img)  # cause we don't need this returned pic, only wanna look at it

    # here we also find eyes on the found picture of a face
    x, y, w, h = faces[0]
    gray_face = gray[y:y+h, x:x+w]  # top left corner is (x,y), bottom right is (x+w,y+h)
    eyes = detect_eyes(gray_face)
    _ = draw_found_objects(eyes, gray_face)  # still don't need the returned pic

    img2 = cv2.imread('obama.jpg')
    gray2 = to_gray(img2)
    faces2 = detect_faces(gray2)
    two_face_swap(img2, faces2)
import time
import os
import cv2
import numpy as np
from PIL import Image


def get_training_data(face_cascade, data_dir):
    images = []
    labels = []
    # os.path.join - joins the paths considering the operating system
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if not f.endswith('.wink')]  # test on wink

    for image_file in image_files:
        img = Image.open(image_file).convert('L')  # open am image & convert it to gray scale using the PIL library
        # convert the image to an numpy array to use in opencv
        img = np.array(img, np.uint8)  # uint8 is an unsigned integer (0 to 255)

        filename = os.path.split(image_file)[1]
        actual_person_number = int(filename.split('.')[0].replace('subject', ''))

        faces = face_cascade.detectMultiScale(img)  # detect faces
        for face in faces:
            x, y, w, h = face
            face_region = img[y:y+h, x:x+w]  # slice the images to only include a face

            # if we gonna use Eigen or Fisher faces, cause they require that training data are of exact same dimension
            # face_region = cv2.resize(face_region, (150, 150))

            images.append(face_region)
            labels.append(actual_person_number)

    return images, labels


def evaluate(recognizer, face_cascade, data_dir):
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wink')]
    num_correct = 0

    for image_file in image_files:
        test_img = Image.open(image_file).convert('L')
        test_img = np.array(test_img, np.uint8)

        # os.path.split - splits the path into to parts, where the second one is the file name (after '/')
        filename = os.path.split(image_file)[1]
        true_person_number = int(filename.split('.')[0].replace('subject', ''))

        faces = face_cascade.detectMultiScale(test_img, 1.05, 6)
        for face in faces:
            x, y, w, h = face
            face_region = test_img[y:y+h, x:x+w]
            # face_region = cv2.resize(face_region, (150, 150))  # for Eigen/Fisher face recognizers

            predicted_person_number, confidence = recognizer.predict(face_region)  # func of the recognizer

            if predicted_person_number == true_person_number:
                print("Correct classified %d with confidence %f" % (true_person_number, confidence))  # %d means integer
                num_correct = num_correct + 1
            else:
                print("Incorrectly classified %d as %d" % (true_person_number, predicted_person_number))

    accuracy = num_correct / float(len(image_files)) * 100
    print("Accuracy: %.2f% %" % accuracy)  # round to 2 decimal places


def predict(recognizer, face_cascade, img):
    predictions = []
    faces = face_cascade.detectMultiScale(img, 1.05, 6)
    for face in faces:
        x, y, w, h = face
        face_region = img[y:y+h, x:x+w]
        # face_region = cv2.resize(face_region, (150, 150))  # for Eigen/Fisher face recognizers
        start = time.time()
        predicted_person_number, confidence = recognizer.predict(face_region)
        print(time.time() - start)
        predictions.append((predicted_person_number, confidence))
    return predictions


def face_rec_in_real_time(face_cascade, face_recognition):
    """ Face recognizer in real time.
    'face_cascade' is needed for face detection on each video's frame,
    'face_recognition' is a pre-trained face recognition model. """
    video_cap = cv2.VideoCapture(0)

    while True:
        # here we take each frame, but for speeding up we can also take each 10th frame, for example
        ret, frame = video_cap.read()
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.05, 6)
        for face in faces:
            x, y, w, h = face
            face_region = gray[y:y+h, x:x+w]

            predicted_person_number, confidence = face_recognition.predict(face_region)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # draws a rectangle around the face
            # to put text on the rectangle: 4th arg is font style, 5th is font scale, 6th is the colour
            cv2.putText(frame, str(predicted_person_number), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

        cv2.imshow('Running face recognition...', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Local Binary Patterns recognizer; can also use Eigen- or FisherFaceRecognizer
    face_recognition = cv2.face.LBPHFaceRecognizer_create()

    print("Getting training examples...")
    images, labels = get_training_data(face_cascade, 'yalefaces')

    print("Training...")
    start = time.time()
    face_recognition.train(images, np.array(labels))  # the function expects a numpy array, not a python list
    print(time.time() - start)
    print("Finished training!")
    evaluate(face_recognition, face_cascade, 'yalefaces')

    # img = cv2.imread('face.jpg', cv2.IMREAD_GRAYSCALE)  # reads the image in the gray scale
    # print predict(face_recognition, face_cascade, img)

    # face_rec_in_real_time(face_cascade, face_recognition)
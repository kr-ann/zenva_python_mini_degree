import numpy as np
from skimage.measure import compare_ssim
import cv2
from filters_app_openCV import to_gray
import smtplib  # for the 'send_gmail' function


def sum_of_pixels(picture=np.ones((3, 3))):
    # if no picture is given, a 3x3 matrix of zeros is used as an example
    """
    s = 0
    for i in range(picture.shape[0]):
        for j in range(picture.shape[1]):
            s += picture[i, j]
    return s
    """
    # a shorter way to do it is
    return picture.sum()


def L1_norm(mat_A=np.array([[0, 230, 75], [0, 210, 60], [0, 200, 50]]),
            mat_B=np.array([[0, 225, 70], [0, 210, 65], [0, 200, 55]])):
    # if no matrices is given, example ones are used
    """
    s = 0
    for i in range(mat_A.shape[0]):
        for j in range(mat_A.shape[1]):
            s += abs(mat_A[i, j] - mat_B[i, j])
    return s
    """
    # a shorter way to do it is
    return abs(mat_A-mat_B).sum()


def SSE(mat_A=np.array([[0, 230, 75], [0, 210, 60], [0, 200, 50]]),
        mat_B=np.array([[0, 225, 70], [0, 210, 65], [0, 200, 55]])):  # smooth function!
    # if matrices are not given, also examples are used
    return ((mat_A-mat_B) ** 2).sum()


def MSE(mat_A=np.ones((10, 10)),
        mat_B=np.zeros((10, 10))):  # robust to an image size!
    # if matrices are not given, also examples are used
    return ((mat_A-mat_B) ** 2).mean()


def SSIM(mat_A=np.ones((10, 10)),
         mat_B=np.zeros((10, 10))):
    # if matrices are not given, also examples are used
    return compare_ssim(mat_A, mat_B, data_range=mat_B.max() - mat_B.min())


def send_gmail(gmail_user, gmail_password, recipient, email_text):
    """ Sends and e-mail via Gmail. """
    # connecting to the server and sending the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()  # insecure connection
        server.starttls()  # upgrade it to a secure one
        server.login(gmail_user, gmail_password)
        server.sendmail(gmail_user, recipient, email_text)
        server.close()
        print('Email is sent!')
    except:
        print('Something went wrong...')
    return


def camera_app():
    """ This is a simple security camera that detects changes in the video and sends you a gmail. """
    cap = cv2.VideoCapture('video.mp4')  # or 'cv2.VideoCapture(0)' to run the video from the webcam

    curr_frame = None
    prev_frame = None
    first_frame = True
    counter = 0
    first_msg = True

    # looking at the video frame by frame
    while True:
        if counter == 0:
            prev_frame = curr_frame

        # we don't need the first returned argument, but it's bool: whether the grabbing of the frame is successful
        _, curr_frame = cap.read()

        if curr_frame is None:  # when we reached the end
            break

        curr_frame = to_gray(curr_frame)  # cause it'll be faster and we don't need color

        if first_frame:  # for the first (and only) case when we don't have any previous frame
            prev_frame = curr_frame
            first_frame = False

        if counter == 9:
            ssim_index = SSIM(curr_frame, prev_frame)
            if ssim_index < 0.8 and first_msg:
                send_gmail('/your_email@gmail.com/',
                           '/your password/',
                           '/recipient\'s_email@gmail.com/',
                           '/text message/')
                first_msg = False
            counter = 0

        # to show the video
        # cv2.imshow('app', curr_frame)

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # closes video file
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    camera_app()

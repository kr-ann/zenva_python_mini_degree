import cv2
import numpy as np


def check_that_loaded(img):
    """ The method checks that the image has been successfully loaded (and prints its shape then). """
    if img is None:
        print("The image hasn't loaded")
    else:
        print("Great!")
        print(img.shape)


def show(img, window='new'):
    """" Shows the input image.
    'window' - the name of the window that's gonna open. """
    cv2.imshow(window, img)  # parameters: 1) name of the window that will pop up, 2) the image
    cv2.waitKey(0)  # it's gonna wait till we press any key
    cv2.destroyAllWindows()  # to delete windows we've created


def to_gray(img):
    """ Converts the image from RGB to the Gray Scale. """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # the second argument is kinda a constant
    return gray


def br_and_cont(img, contrast, brightness):
    """ Applies brightness and contrast to the image. """
    # cv2 will turn the brightness value into a matrix
    cb_img = cv2.addWeighted(img, contrast, np.zeros(img.shape, dtype=img.dtype), 0, brightness)  # weighted sum
    # check the equation in the notes: matrix of zeros and "0" form the second member of sum (that we don't need here)
    # we can also use img.copy() instead of zero matrix, but it's less efficient
    return cb_img


def apply_convolution(img, kernel_num):
    """ Applies convolution taking the image and the number of the kernel.
    'kernel_num' - what kernel to apply. """

    # identity kernel: returns the same image
    kernel_0 = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    # sharpen kernel (~резкость)
    kernel_1 = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # blurring
    kernel_2 = cv2.getGaussianKernel(3, 0)
    kernel_3 = cv2.getGaussianKernel(7, 0)

    # box_kernel, also blurring
    kernel_4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32) / 9

    # random kernel
    kernel_5 = np.array([
        [5, -2, 5],
        [-4, 0, -4],
        [5, -2, 5]
    ])

    kernels = [kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5]

    if isinstance(kernel_num, int) and kernel_num in range(len(kernels)):
        kernel = kernels[kernel_num]
    else:
        raise ValueError("No such option. ")

    # the 2nd argument is depth (number of color channels), "-1" - do the convolution for the entire depth of the image
    convolved = cv2.filter2D(img, -1, kernel)
    return convolved


def dummy(val):
    """ The func that doesn't do anything.
    It'll be called when values of our track bars change, cause we don't wanna do anything then. """
    pass


def the_app(color_original):
    """ Creates the app where we can apply effects to a picture. """
    color_modified = color_original.copy()
    gray_original = to_gray(color_original)
    gray_modified = gray_original.copy()

    cv2.namedWindow('app')
    cv2.createTrackbar('contrast', 'app', 1, 25, dummy)  # '1' - initial, '100' - max value, dummy - 'onChange'
    cv2.createTrackbar('brightness', 'app', 50, 100, dummy)  # '50' is the new zero, so we can subtract brightness too
    cv2.createTrackbar('filters', 'app', 0, 5, dummy)  # '5' is 'len(kernels)-1' from the 'apply_convolution' function
    cv2.createTrackbar('grayscale', 'app', 0, 1, dummy)  # '0' and '1' cause it's a switch

    counter = 1

    while True:  # UI loop
        grayscale = cv2.getTrackbarPos('grayscale', 'app')
        if grayscale == 0:  # we decide what picture to show depending on the grayscale bar value
            pic_to_show = color_modified
        else:
            pic_to_show = gray_modified
        cv2.imshow('app', pic_to_show)

        key = cv2.waitKey(1) & 0xFF  # what is this? check the next comment
        """ 
        0xFF is a hexadecimal constant. By using bitwise_AND (&) with this constant, it leaves only the last 8 bits 
        of the original.
        ord('q') can return different numbers if you have NumLock activated. For example, when pressing c, the code:
        / print(cv2.waitKey(10)) / returns 1048675 when NumLock is activated and 99 otherwise.
        Converting these 2 numbers to binary:
        1048675 = 100000000000001100011
        99 = 1100011
        The last byte is identical, so we take just this last byte (the rest is caused because of the state of NumLock). 
        """
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('output%d.png' % counter, pic_to_show)
            counter += 1

        contrast = cv2.getTrackbarPos('contrast', 'app')
        brightness = cv2.getTrackbarPos('brightness', 'app')
        kernel_num = cv2.getTrackbarPos('filters', 'app')

        color_modified = apply_convolution(color_original, kernel_num)  # we apply filters first
        # we have to copies: one gray and one color, cause we can't convert from Gray to RGB
        gray_modified = apply_convolution(gray_original, kernel_num)
        # 'brightness-50' in the next line cause we want 50 to be a zero value
        color_modified = br_and_cont(color_modified, contrast, brightness-50)
        gray_modified = br_and_cont(gray_modified, contrast, brightness - 50)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    pic = cv2.imread("test.jpg")  # load an image
    the_app(pic)
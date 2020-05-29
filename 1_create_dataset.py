"""
title   :   1_create_dataset.py
author  :   Kamal Lal (www.linkedin.com/in/kamal-lal-40671188/)
date    :   May 2020
version :   0.1
python  :   3.7

notes:
  - Purpose of this module is to create image dataset of various hand positions.
  - Creates 250 jpg files each.

usage:
  - Run the module.
  - Place hand in front of camera such that the red box contains ONLY then hand.
  - Press 's' in keyboard to save the hand histogram.
  - The ROI window shows the background subtracted feed containing only the hand.
  - If unclear, press 'r' to reset the saved histogram and repeat previous three steps.
  - Show hand gesture for 0 (fist) to camera, placing it in the green box and press '0'
    to save 250 jpg files in relavant folder. Repeat this for 1, 2, 3, 4 and 5.
  - Show some 'invalid' jestures and save them by pressing '6'.
  - Press ESC to exit.

"""

# built-in modules import
import os
import time

# external modules import
import cv2


IMGS_REQUIRED_COUNT = 250
CAPTURE_INTERVAL = 0.2


def save_imgs(key_pressed, img_to_save):
    """
    Function to save image (roi) to relavant folder if valid key is pressed.

    Args:
        key_pressed: ASCII value of the key pressed.

        img_to_save: ROI of the background subtracted image.

    Returns:
        bool: True if less than 250 images saved.
              False if invalid key pressed or if 250 images are saved.

    """

    if ord('0') <= key_pressed <= ord('6'):
        if key_pressed == ord('6'):
            base_path = os.path.join(os.getcwd(), 'training_imgs', 'none')
        else:
            base_path = os.path.join(os.getcwd(), 'training_imgs', chr(key_pressed))

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        pic_num = len(os.listdir(base_path)) + 1

        if pic_num <= IMGS_REQUIRED_COUNT:
            img_name = f'img_{pic_num:03d}.jpg'
            path = os.path.join(base_path, img_name)
            cv2.imwrite(path, img_to_save)
        else:
            return True
    return False


def calc_histogram(img_roi):
    """
    Function to calculate the histogram of given image.

    Args:
        img_roi: ROI of the image to calucate histogram.

    Returns:
        list: The calculated histogram.

    """

    roi_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist


def remove_bg(img_with_bg, hist):
    """
    Function to remove background and filter 'hand'.

    Args:
        img_with_bg: ROI of the image from which background need to be removed.

        hist: Histogram of the hand.

    Returns:
        image: The background subtracted image

    """

    roi_hsv = cv2.cvtColor(img_with_bg, cv2.COLOR_BGR2HSV)

    roi_projected = cv2.calcBackProject([roi_hsv], [0, 1], hist, [0, 180, 0, 256], scale=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(roi_projected, -1, kernel, roi_projected)

    mask = cv2.threshold(roi_projected, 120, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.merge((mask, mask, mask))

    return cv2.bitwise_and(img_with_bg, mask)


# initializations
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

x1, y1, h1, w1 = 50, 50, 200, 120
rect_color = (26, 71, 255)
hand_hist = None
last_key = None

# main loop
while cam.isOpened():
    _ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    roi = frame[y1:y1 + h1, x1:x1 + w1].copy()

    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), rect_color, 2)

    if hand_hist is not None:
        roi = remove_bg(roi, hand_hist)

    cv2.imshow('Camera', frame)
    cv2.imshow('ROI', roi)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        # press ESC to exit
        if key == 27:
            break

        # press 's' to save hand histogram
        if key == ord('s'):
            hand_hist = calc_histogram(roi)
            w1 = 200
            rect_color = (0, 255, 0)

        # press 'r' to reset the saved hand histogram
        if key == ord('r'):
            hand_hist = None
            w1 = 120
            rect_color = (26, 71, 255)

        # press '0' to '6' to save image to corresponding folder
        if ord('0') <= key <= ord('6'):
            last_key = key

    if last_key is not None:
        count_reached = save_imgs(last_key, roi)
        time.sleep(CAPTURE_INTERVAL)
        if count_reached:
            last_key = None

cam.release()
cv2.destroyAllWindows()

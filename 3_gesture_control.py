"""
title   :   3_gesture_control.py
author  :   Kamal Lal (www.linkedin.com/in/kamal-lal-40671188/)
date    :   May 2020
version :   0.1
python  :   3.7

notes:
  - This module uses the trained neural network model to detect 'hand gesture' from
    camera feed and send relavant commands to WinCC over OPC UA.

usage:
  - Run the module.
  - Place hand in front of camera such that the red box contains ONLY then hand.
  - Press 's' in keyboard to save the hand histogram.
  - The ROI window shows the background subtracted feed containing only the hand.
  - If unclear, press 'r' to reset the saved histogram and repeat previous three steps.
  - Show hand gesture to camera, placing it in the green box.
  - WinCC logic uses a 'stabilized' variant of prediction and send commands.
  - Press ESC to exit.

"""

# built-in modules import
import time

# external modules import
import cv2
import numpy as np
import tensorflow as tf
from opcua import ua, Server


MODEL_PATH = 'gest_recog_model.h5'
IMG_SIZE = 50
STABILIZATION_TIME = 0.5
LABELS = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'none']


def start_opc():
    """
    The function initializes an OPC server and creates required tags for WinCC.

    Args:
        (nil)

    Returns:
        tags: A list of OPC tags to be sent to WinCC.

        server: The initialised server itself is returned. Will be later used
                to stop the server.

    """

    # creates a 'local' server at port '8020'
    server = Server()
    server.set_endpoint("opc.tcp://127.0.0.1:8020/")

    uri = "gesture_control"
    idx = server.register_namespace(uri)
    objects = server.get_objects_node()
    obj = objects.add_object(idx, "GestureTest")

    # fan-1 tags
    a_sel = obj.add_variable(idx, "f1_sel", False)
    a_cmd = obj.add_variable(idx, "f1_cmd", False)
    a_sel.set_writable()
    a_cmd.set_writable()

    # fan-2 tags
    b_sel = obj.add_variable(idx, "f2_sel", False)
    b_cmd = obj.add_variable(idx, "f2_cmd", False)
    b_sel.set_writable()
    b_cmd.set_writable()

    # fan-3 tags
    c_sel = obj.add_variable(idx, "f3_sel", False)
    c_cmd = obj.add_variable(idx, "f3_cmd", False)
    c_sel.set_writable()
    c_cmd.set_writable()

    tags = [a_sel, a_cmd, b_sel, b_cmd, c_sel, c_cmd]

    # start server
    server.start()

    return tags, server


def stop_opc(opc_server):
    """
    The function stops the OPC server.

    Args:
        (nil)

    Returns:
        (nil)

    """
    opc_server.stop()


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


def stabilize(new_prediction, last_prediction, prev_stable_val):
    """
    Function to stabilize prediction value.

    Args:
        new_prediction: Current prediction value.

        last_prediction: Last prediction value.

        prev_stable_val: Last stable prediction value.

    Returns:
        stabilized prediction: Returns new prediction if it is equal to last_prediction
                               for a duration. Else returns the last stable value.

    """

    global start, end

    if new_prediction != last_prediction:
        start = time.time()

    if new_prediction == last_prediction:
        end = time.time()
    else:
        end = start

    if end - start > STABILIZATION_TIME:
        return new_prediction

    return prev_stable_val


def send_to_wincc(stable_val, opc_tags):
    """
    Function to send data to WinCC.

    Args:
        stable_val: The 'stabilised' prediction value.

        opc_tags: List of OPC tags.

    Returns:
        (nil)

    """

    global selection, fist, palm
    p1_sel, p1_cmd, p2_sel, p2_cmd, p3_sel, p3_cmd = opc_tags

    # selection logic
    if stable_val == 1:
        p1_sel.set_value(True, ua.VariantType.Boolean)
        p2_sel.set_value(False, ua.VariantType.Boolean)
        p3_sel.set_value(False, ua.VariantType.Boolean)
        selection = 1
    elif stable_val == 2:
        p1_sel.set_value(False, ua.VariantType.Boolean)
        p2_sel.set_value(True, ua.VariantType.Boolean)
        p3_sel.set_value(False, ua.VariantType.Boolean)
        selection = 2
    elif stable_val == 3:
        p1_sel.set_value(False, ua.VariantType.Boolean)
        p2_sel.set_value(False, ua.VariantType.Boolean)
        p3_sel.set_value(True, ua.VariantType.Boolean)
        selection = 3

    # on/off logic
    if 1 <= selection <= 3:
        if stable_val == 0:
            fist = True
        if stable_val == 4 or stable_val == 5:
            palm = True
        if stable_val > 5:
            fist = False
            palm = False

        # palm open gesture
        if fist and (stable_val == 4 or stable_val == 5):
            fist = False
            if selection == 1:
                p1_cmd.set_value(True, ua.VariantType.Boolean)
            if selection == 2:
                p2_cmd.set_value(True, ua.VariantType.Boolean)
            if selection == 3:
                p3_cmd.set_value(True, ua.VariantType.Boolean)

        # palm closing gesture
        if palm and stable_val == 0:
            palm = False
            if selection == 1:
                p1_cmd.set_value(False, ua.VariantType.Boolean)
            if selection == 2:
                p2_cmd.set_value(False, ua.VariantType.Boolean)
            if selection == 3:
                p3_cmd.set_value(False, ua.VariantType.Boolean)


# initializations
selection = 0
fist = palm = False
start = end = time.time()
last_n = val = 6

x1, y1, h1, w1 = 50, 50, 200, 120
rect_color = (26, 71, 255)
hand_hist = None

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# create OPC server and tags
wincc_tags, srvr = start_opc()

# main loop
while cam.isOpened():
    _ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    roi = frame[y1:y1 + h1, x1:x1 + w1].copy()

    cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), rect_color, 2)

    if hand_hist is not None:
        roi = remove_bg(roi, hand_hist)

        frame_to_predict = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frame_to_predict = cv2.resize(frame_to_predict, (IMG_SIZE, IMG_SIZE))
        frame_to_predict = np.reshape(frame_to_predict, (-1, IMG_SIZE, IMG_SIZE, 1))

        # make prediction and get stabilized val
        model_prediction = model.predict(frame_to_predict)
        n = np.argmax(model_prediction)
        val = stabilize(n, last_n, val)

        send_to_wincc(val, wincc_tags)
        last_n = n

        cv2.putText(roi, LABELS[int(val)], (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow('Camera', frame[0:y1+y1+h1, 0:x1+x1+w1])
        # cv2.imshow('Camera', frame)
    else:
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

# stop OPC server
stop_opc(srvr)

cam.release()
cv2.destroyAllWindows()

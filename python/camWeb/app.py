from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, abort
import subprocess
from shlex import quote
import json
import numpy as np
import cv2
import time
import serial
import socket
import pickle
import psutil
import subprocess
import re
import math
from scipy.spatial import distance as dist
from typing import Union, Tuple, Any, Optional
import imutils
from imutils import perspective
from imutils import contours
import argparse
import pandas as pd
import locale
from collections import deque
from imutils.video import VideoStream
import threading

locale.setlocale(locale.LC_ALL, '')

app = Flask(__name__)

running_video = False
running_shape = False
running_color = False
running_size = False
running_calibration = False
running_frame = False
running_white = False
running_roundness = False
running_detect_obj = False
running_aruco = False
running_characteristic = False
running_detect_line = False
running_exp = False
running_ssd = False
running_FPV = False
running_tracking = False

disabled_button = 'disabled="disabled"'

camera = cv2.VideoCapture(cv2.CAP_V4L2)
FPS = camera.get(cv2.CAP_PROP_FPS)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
pts = deque(maxlen=32)

#path_prefix = "/code/python/camWeb"
path_prefix = "./"

classNames = []
classFile = path_prefix + "/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = path_prefix + "/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = path_prefix + "/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

pwm_data = [0, 0, 0, 0]
pwm_data_track = [0, 0]

X = -1
Y = -1

white_balance = 0
blur = 0
sharp = 0
noise = 0
LIMUN = 0

HMIN = 50
SMIN = 50
VMIN = 50
HMAX = 255
SMAX = 255
VMAX = 255
AREA = 40000

use_serial = False
running_video = False
camera_available = True

index = ["color", "color_name", "hex", "R", "G", "B"]
#csv = pd.read_csv('/code/python/camWeb/colors.csv', names=index, header=None)
csv = pd.read_csv('colors.csv', names=index, header=None)

start_time = time.time()
display_time = 2
fc = 0
FPS_new = 0

try:
    ser = serial.Serial("/dev/ttyS0", 9600, timeout=10)
    ser1 = serial.Serial("/dev/ttyS1", 9600, timeout=10)
    ser.flush()
    use_serial = True
except OSError as e:
    print("Низкое питание на плате!!!")
    use_serial = False

if not camera.isOpened():
    print("Камера не запущена")
    camera_available = False

rvecs, tvecs = None, None
cv_file = cv2.FileStorage("calibration_matrix.yaml", cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_matrix = cv_file.getNode("dist_coeff").mat()
cv_file.release()


def send_detected():
    global use_serial, X, Y
    while True:
        #print(X)
        #print(Y)
        if use_serial:
            ser1.write((str(X) + " " + str(Y) + str('\n')).encode())


send_detected = threading.Thread(target=send_detected)
send_detected.daemon = True
send_detected.start()


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshConstant = 10
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if np.all(ids != None):
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_matrix)
        for i in range(0, ids.size):
            cv2.drawFrameAxes(frame, camera_matrix, dist_matrix, rvec[i], tvec[i], 0.1)
        cv2.aruco.drawDetectedMarkers(frame, corners)
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0]) + ', '

        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def calibrationCam(frame, FPS, width, height):
    image = frame
    CHECKERBOARD = (9, 6)
    MIN_POINTS = 50
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    threedpoints = []
    twodpoints = []
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH
                                             + cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        threedpoints.append(objectp3d)
        corners2 = cv2.cornerSubPix(
            gray, corners, CHECKERBOARD, (-1, -1), criteria)
        twodpoints.append(corners2)
        image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
    if len(threedpoints) > 0:
        h, w = image.shape[:2]
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None,
                                                                      None)
        print(" Camera matrix:")
        print(matrix)
        print("\n Distortion coefficient:")
        print(distortion)
        print("\n Rotation Vectors:")
        print(r_vecs)
        print("\n Translation Vectors:")
        print(t_vecs)
        """
        f = open('calibration.pckl', 'wb')
        pickle.dump((matrix, distortion, r_vecs, t_vecs ), f)
        f.close()
        """
        cv_file = cv2.FileStorage("calibration_matrix.yaml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("camera_matrix", matrix)
        cv_file.write("dist_coeff", distortion)

    return image


def character(image):
    global start_time, display_time, fc, FPS_new
    fc += 1
    TIME = time.time() - start_time
    if (TIME) >= display_time:
        FPS_new = fc / (TIME)
        fc = 0
        start_time = time.time()
    fps_disp = "FPS: " + str(FPS_new)[0:5]
    image = cv2.putText(image, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
    return image


def check_CPU_temp():
    temp = None
    err, msg = subprocess.getstatusoutput('cat /sys/class/thermal/thermal_zone0/temp')
    if not err:
        m = re.search(r'-?\d\.?\d*', msg)  # a solution with a  regex
        try:
            temp = float(m.group()) / 1000
        except ValueError:  # catch only error needed
            pass
    return temp


def check_RAM_usage():
    return round(psutil.virtual_memory()[3] / 1000000, 2)


def check_CPU_usage():
    return psutil.cpu_percent(4)


def color_detect(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    hsvmask = cv2.inRange(hsvFrame, lower_range, upper_range)

    redLower = (0, 47, 85)
    redUpper = (22, 255, 255)

    redLower1 = (157, 48, 75)
    redUpper1 = (179, 255, 255)

    redmask1 = cv2.inRange(hsvFrame, redLower, redUpper)
    redmask2 = cv2.inRange(hsvFrame, redLower1, redUpper1)
    red_minus = redmask1 + redmask2

    greenLower = (42, 80, 41)
    greenUpper = (81, 255, 255)
    green_minus = cv2.inRange(hsvFrame, greenLower, greenUpper)

    blueLower = (86, 53, 0)
    blueUpper = (129, 255, 255)
    blue_minus = cv2.inRange(hsvFrame, blueLower, blueUpper)

    red_mask = red_minus
    green_mask = green_minus
    blue_mask = blue_minus

    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)

    blue_mask = cv2.erode(blue_mask, None, iterations=2)
    blue_mask = cv2.dilate(blue_mask, None, iterations=2)

    green_mask = cv2.erode(green_mask, None, iterations=2)
    green_mask = cv2.dilate(green_mask, None, iterations=2)
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500 and area < AREA:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255), 2)

            cv2.putText(frame, "Красный", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 1.0,
                        (0, 0, 255))
    contours, hierarchy = cv2.findContours(green_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500 and area < AREA:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0), 2)

            cv2.putText(frame, "Зеленый", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.0, (0, 255, 0))
    contours, hierarchy = cv2.findContours(blue_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500 and area < AREA:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y),
                                  (x + w, y + h),
                                  (255, 0, 0), 2)

            cv2.putText(frame, "Синий", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.0, (255, 0, 0))
    return frame


def bluring_fun(image, b):
    if b == 0:
        return image
    else:
        blurred = cv2.blur(image, (b, b))
        return blurred


def noising(image, n):
    if n == 0:
        return image
    else:
        gn_img = cv2.GaussianBlur(image, (15, 15), n, n, cv2.BORDER_CONSTANT)
        return gn_img


def sharping_fun(image, s):
    if s == 0:
        return image
    else:
        sharp_image = cv2.filter2D(image, -1, (s, s))
        return sharp_image


def exp(image, b, s, n):
    bluring = bluring_fun(image, b)
    noiseness = noising(bluring, n)
    sharping = sharping_fun(noiseness, s)
    result = sharping
    return result


def _map(x, in_min, in_max, out_min, out_max):
    val = int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
    return val if val > 0 else 0


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def getCurvature(contour, stride=1):
    curvature = []
    assert stride < len(contour), "stride must be shorther than length of contour"

    for i in range(len(contour)):
        before = i - stride + len(contour) if i - stride < 0 else i - stride
        after = i + stride - len(contour) if i + stride >= len(contour) else i + stride
        f1x, f1y = (contour[after] - contour[before]) / stride
        f2x, f2y = (contour[after] - 2 * contour[i] + contour[before]) / stride ** 2
        denominator = (f1x ** 2 + f1y ** 2) ** 3 + 1e-11
        curvature_at_i = np.sqrt(4 * (f2y * f1x - f2x * f1y) ** 2 / denominator) if denominator > 1e-12 else -1
        curvature.append(curvature_at_i)
    return curvature


def val_contrast(img, LIMUN):
    if LIMUN == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(LIMUN, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


def getPerpCoord(aX, aY, bX, bY, length):
    vX = bX - aX
    vY = bY - aY
    if (vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX * vX + vY * vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0 - vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)


def lineDet(img, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, LIMUN):
    img = val_contrast(img, LIMUN)
    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    hsv = img
    gray = cv2.inRange(hsv, lower_range, upper_range)
    edges = cv2.Canny(gray, 0, 500, 5)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    try:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            D = (dist.euclidean((x1, y1), (x2, y2)))
            (mX, mY) = midpoint((x1, y1), (x2, y2))
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.line(img, (int(mX), int(mY)), (320, 240), (0, 255, 0), 2)
            cv2.putText(img, "{:.1f}".format(D), (int(mX + 20), int(mY - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            D1 = (dist.euclidean((320, 240), (mX, mY)))
            cv2.putText(img, "{:.1f}".format(D1), (int(mX - 20), int(mY + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    except Exception as e:
        pass

    try:
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if (area < AREA) and (area > 500):
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                if len(approx) > 10:
                    M = cv2.moments(cnt)
                    ((x_axis, y_axis), radius) = cv2.minEnclosingCircle(cnt)
                    center = (int(x_axis), int(y_axis))
                    radius = int(radius)
                    xA = int(M["m10"] / M["m00"])
                    yA = int(M["m01"] / M["m00"])
                    xB = 320
                    yB = 240
                    cv2.putText(img, str(radius), center, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawContours(img, [cnt], 0, (0, 0, 0), 2)
                    D = (dist.euclidean((xA, yA), (xB, yB)))
                    (mX, mY) = midpoint((xA, yA), (xB, yB))
                    cv2.line(img, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
                    cv2.putText(img, "{:.1f}".format(D), (int(mX), int(mY - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    except Exception as e:
        print(e)

    return img


def detect_distance_between_obj(image, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA):
    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    hsv = image  # cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    edged = cv2.Canny(mask, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
                  (255, 0, 255))
        refObj = None
        for c in cnts:
            area = cv2.contourArea(c)
            if (area < AREA) and (area > 500):
                cv2.drawContours(image, c, -1, (0), 5)
                if cv2.contourArea(c) < 100:
                    continue
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                cX = np.average(box[:, 0])
                cY = np.average(box[:, 1])
                if refObj is None:
                    (tl, tr, br, bl) = box
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)
                    D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    refObj = (box, (cX, cY), D / 1920)
                    continue
                cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
                cv2.drawContours(image, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
                refCoords = np.vstack([refObj[0], refObj[1]])
                objCoords = np.vstack([box, (cX, cY)])
                xA = cX
                yA = cY
                xB = refObj[1][0]
                yB = refObj[1][1]
                color = colors[1]
                cv2.circle(image, (int(xA), int(yA)), 5, color, -1)
                cv2.circle(image, (int(xB), int(yB)), 5, color, -1)
                cv2.line(image, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)
                D = (dist.euclidean((xA, yA), (xB, yB)) / refObj[2])
                (mX, mY) = midpoint((xA, yA), (xB, yB))
                cv2.putText(image, "{:.1f}".format(D), (int(mX), int(mY - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return image


def round_contr(image, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA):
    hsvFrame = image
    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    mask = cv2.inRange(hsvFrame, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_COMPLEX
    for cnt in contours:
        if cv2.contourArea(cnt) < 500 or cv2.contourArea(cnt) > AREA:
            continue
        (x_axis, y_axis), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x_axis), int(y_axis))
        radius = int(radius)
        image = cv2.putText(image, str(radius), center, font, 1,
                            (0, 0, 0), 2, cv2.LINE_AA, False)
        image = cv2.circle(image, center, radius, (0, 255, 0), 2)
    return image


def get_area(c, frame):
    pixelsPerMetric = None
    global AREA
    if cv2.contourArea(c) < 500 or cv2.contourArea(c) > AREA:
        return frame
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    for (x, y) in box:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 360
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    cv2.putText(frame, "{:.1f}".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(frame, "{:.1f}".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    return frame


def getColorName(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


def shape_call(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA):
    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    hsv = frame
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    used_contours = []
    font = cv2.FONT_HERSHEY_COMPLEX
    mlist = []
    clist = []
    ks = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 500 or cv2.contourArea(cnt) > AREA:
            continue
        coloring = ""

        try:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            b, g, r = frame[cY, cX]
            b = int(b)
            g = int(g)
            r = int(r)

            # Creating text string to display( Color name and RGB values )
            coloring = getColorName(r, g, b)


        except Exception as e:
            pass

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], 0, (0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx) == 3:
            cv2.putText(frame, coloring + " Треугольник", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

        elif len(approx) == 4:
            x2 = approx.ravel()[2]
            y2 = approx.ravel()[3]
            x4 = approx.ravel()[6]
            y4 = approx.ravel()[7]
            side1 = abs(x2 - x)
            side2 = abs(y4 - y)

            if abs(side1 - side2) <= 1:
                cv2.putText(frame, coloring + " Квадрат", (x, (y - 5)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
            else:
                cv2.putText(frame, coloring + " Прямоугольник", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

        elif len(approx) > 10:
            cv2.putText(frame, coloring + " Круг", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def size_detect(image, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA):
    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    hsv = image
    mask = cv2.inRange(hsv, lower_range, upper_range)
    gray = cv2.GaussianBlur(mask, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    orig = image.copy()
    if len(cnts) > 0:
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None
        for c in cnts:
            if cv2.contourArea(c) < 500 or cv2.contourArea(c) > AREA:
                continue
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / 360
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            cv2.putText(orig, "{:.1f}".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
    return orig


def ssd(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    # print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img


def track_ob(frame, pwm_data_track):
    global pts, X, Y
    counter = 0
    (dX, dY) = (0, 0)
    direction = ""
    frame = imutils.resize(frame, width=360)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    greenLower = (1, 100, 197)
    greenUpper = (13, 255, 255)

    greenLower1 = (164, 99, 198)
    greenUpper1 = (179, 255, 255)

    mask1 = cv2.inRange(hsv, greenLower, greenUpper)
    mask2 = cv2.inRange(hsv, greenLower1, greenUpper1)
    mask = mask1 + mask2

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            pts.appendleft(center)
        if len(pts) >= 6:
            for i in np.arange(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                if counter >= 10 and i == 1 and pts[-5] is not None:
                    dX = pts[-5][0] - pts[i][0]
                    dY = pts[-5][1] - pts[i][1]
                    (dirX, dirY) = ("", "")
                    if np.abs(dX) > 20:
                        dirX = "East" if np.sign(dX) == 1 else "West"
                    if np.abs(dY) > 20:
                        dirY = "North" if np.sign(dY) == 1 else "South"
                    if dirX != "" and dirY != "":
                        direction = "{}-{}".format(dirY, dirX)
                    else:
                        direction = dirX if dirX != "" else dirY
                Y = pts[0][1]
                X = pts[0][0]
                thickness = int(np.sqrt(32 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 255), 3)
                cv2.putText(frame, "dx: {}, dy: {}".format(pts[0][0], pts[0][1]),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 255), 1)
                pwm_data_track = [pts[0][0], pts[0][1]]

    return frame, pwm_data_track


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def white_balancing(image, val):
    original = image
    gamma = val
    if not val == 0:
        adjusted = adjust_gamma(original, gamma=gamma)
        return adjusted
    else:
        return original


@app.route('/')
def index():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking, camera_available
    running_video = True
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    if camera_available == False:
        return render_template('camera.html'), 404
    return render_template('index.html', disabled=disabled_button)


@app.route('/update_variable', methods=['POST'])
def update_variable():
    global disabled_button, blur, sharp, noise, white_balance

    disabled_button = 'disabled="disabled"'

    if request.method == 'POST':
        blur, sharp, noise, white_balance = 0, 0, 0, 0

    return "OK", 200


@app.route("/color_detect", methods=['GET', 'POST'])
def color_detect_page():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_tracking, \
        HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, running_FPV
    if request.method == 'POST':
        if (request.json.get("H_max") is not None):
            HMAX = request.json.get("H_max")
            print(HMAX)
        if (request.json.get("H_min") is not None):
            HMIN = request.json.get("H_min")
        if (request.json.get("S_max") is not None):
            SMAX = request.json.get("S_max")
        if (request.json.get("S_min") is not None):
            SMIN = request.json.get("S_min")
        if (request.json.get("V_max") is not None):
            VMAX = request.json.get("V_max")
        if (request.json.get("V_min") is not None):
            VMIN = request.json.get("V_min")
        if (request.json.get("Area") is not None):
            AREA = request.json.get("Area")
    running_video = False
    running_color = True
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_def_of_color.html', disabled=disabled_button)


@app.route('/size_detect', methods=['GET', 'POST'])
def size_detect_page():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_tracking, \
        HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, running_FPV
    if request.method == 'POST':
        if (request.json.get("H_max") is not None):
            HMAX = request.json.get("H_max")
            print(HMAX)
        if (request.json.get("H_min") is not None):
            HMIN = request.json.get("H_min")
        if (request.json.get("S_max") is not None):
            SMAX = request.json.get("S_max")
        if (request.json.get("S_min") is not None):
            SMIN = request.json.get("S_min")
        if (request.json.get("V_max") is not None):
            VMAX = request.json.get("V_max")
        if (request.json.get("V_min") is not None):
            VMIN = request.json.get("V_min")
        if (request.json.get("Area") is not None):
            AREA = request.json.get("Area")
    running_video = False
    running_color = False
    running_size = True
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_s_kontur.html', disabled=disabled_button)


@app.route('/calibration', methods=['GET', 'POST'])
def calibration():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking
    running_video = False
    running_color = False
    running_size = False
    running_calibration = True
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    if request.method == 'POST':
        if request.form['calibration_button'] == 'Calibration':
            pass
    elif request.method == 'GET':
        return render_template('page_calibration.html', disabled=disabled_button)


@app.route('/shape_classification', methods=['GET', 'POST'])
def shape_detect():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_tracking, \
        HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, running_FPV
    if request.method == 'POST':
        if (request.json.get("H_max") is not None):
            HMAX = request.json.get("H_max")
            print(HMAX)
        if (request.json.get("H_min") is not None):
            HMIN = request.json.get("H_min")
        if (request.json.get("S_max") is not None):
            SMAX = request.json.get("S_max")
        if (request.json.get("S_min") is not None):
            SMIN = request.json.get("S_min")
        if (request.json.get("V_max") is not None):
            VMAX = request.json.get("V_max")
        if (request.json.get("V_min") is not None):
            VMIN = request.json.get("V_min")
        if (request.json.get("Area") is not None):
            AREA = request.json.get("Area")
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = True
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_figures.html', disabled=disabled_button)


@app.route('/white_balance', methods=['GET', 'POST'])
def white_balance_page():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, white_balance, running_FPV, running_tracking, disabled_button
    if request.method == 'POST':
        if (request.json.get("val") is not None):
            white_balance = request.json.get("val")
            disabled_button = ""

    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = True
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_bwhite.html', val=white_balance, disabled=disabled_button)


@app.route('/roundness', methods=['GET', 'POST'])
def roundness():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_tracking, \
        HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, running_FPV
    if request.method == 'POST':
        if (request.json.get("H_max") is not None):
            HMAX = request.json.get("H_max")
            print(HMAX)
        if (request.json.get("H_min") is not None):
            HMIN = request.json.get("H_min")
        if (request.json.get("S_max") is not None):
            SMAX = request.json.get("S_max")
        if (request.json.get("S_min") is not None):
            SMIN = request.json.get("S_min")
        if (request.json.get("V_max") is not None):
            VMAX = request.json.get("V_max")
        if (request.json.get("V_min") is not None):
            VMIN = request.json.get("V_min")
        if (request.json.get("Area") is not None):
            AREA = request.json.get("Area")
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = True
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_round_kontur.html', disabled=disabled_button)


@app.route('/detect_dist', methods=['GET', 'POST'])
def detect_destination():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_tracking, \
        HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, running_FPV
    if request.method == 'POST':
        if (request.json.get("H_max") is not None):
            HMAX = request.json.get("H_max")
            print(HMAX)
        if (request.json.get("H_min") is not None):
            HMIN = request.json.get("H_min")
        if (request.json.get("S_max") is not None):
            SMAX = request.json.get("S_max")
        if (request.json.get("S_min") is not None):
            SMIN = request.json.get("S_min")
        if (request.json.get("V_max") is not None):
            VMAX = request.json.get("V_max")
        if (request.json.get("V_min") is not None):
            VMIN = request.json.get("V_min")
        if (request.json.get("Area") is not None):
            AREA = request.json.get("Area")
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = True
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_where_object.html', disabled=disabled_button)


@app.route('/aruco')
def aruco():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = True
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_aruko_markers.html', disabled=disabled_button)


@app.route('/charact')
def charact():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = True
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    cpu_temp = check_CPU_temp()
    cpu_usage = check_CPU_usage()
    ram_usage = check_RAM_usage()
    IPAddr = ((([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [
        [(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in
         [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0])
    address = "https://" + str(IPAddr) + ":6175"
    return render_template('page_chars_of_camera.html', cpu_temp=cpu_temp, cpu_usage=cpu_usage, ram_usage=ram_usage,
                           address=address)


@app.route('/send', methods=['GET', 'POST'])
def send():
    try:
        if (request.form['command']):
            stdout, stderr = subprocess.getstatusoutput(request.form['command'])
        else:
            stdout, stderr = (b"command not found", b"")

        data = {}
        data['command'] = request.form['command']
        data['data'] = request.form['data']
        data['result'] = stdout + "\n" + stderr
        return (json.dumps(data))
    except Exception as e:
        print(e)


@app.route('/line_detect', methods=['GET', 'POST'])
def line_detect():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_tracking, \
        HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, running_FPV, LIMUN
    if request.method == 'POST':
        if (request.json.get("H_max") is not None):
            HMAX = request.json.get("H_max")
            print(HMAX)
        if (request.json.get("H_min") is not None):
            HMIN = request.json.get("H_min")
        if (request.json.get("S_max") is not None):
            SMAX = request.json.get("S_max")
        if (request.json.get("S_min") is not None):
            SMIN = request.json.get("S_min")
        if (request.json.get("V_max") is not None):
            VMAX = request.json.get("V_max")
        if (request.json.get("V_min") is not None):
            VMIN = request.json.get("V_min")
        if (request.json.get("Area") is not None):
            AREA = request.json.get("Area")
        if (request.json.get("Luminous") is not None):
            LIMUN = request.json.get("Luminous")
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = True
    running_exp = False
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_lines.html', val8=LIMUN, disabled=disabled_button)


@app.route('/exposition', methods=['GET', 'POST'])
def expos():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking, \
        blur, sharp, noise, white_balance, disabled_button
    if request.method == 'POST':
        if (request.json.get("blur") is not None):
            blur = request.json.get("blur")
            # print(blur)
        if (request.json.get("sharp") is not None):
            sharp = request.json.get("sharp")
            # print(sharp)
        if (request.json.get("noise") is not None):
            noise = request.json.get("noise")
            # print(noise)
        if (request.json.get("white_balance") is not None):
            white_balance = request.json.get("white_balance")
        disabled_button = ""
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = True
    running_ssd = False
    running_FPV = False
    running_tracking = False
    return render_template('page_expo.html', val1=blur, val2=sharp, val3=noise, val4=white_balance,
                           disabled=disabled_button)


@app.route('/ssd')
def ssd_webpage():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_ssd = True
    running_FPV = False
    running_tracking = False
    return render_template('page_recog_ai.html', disabled=disabled_button)


@app.route('/sendFPV', methods=['GET', 'POST'])
def data():
    global pwm_data, use_serial, ser
    if request.method == 'POST':
        data_json = request.get_json()
        joy1X = data_json["joy1X"]
        joy1Y = data_json["joy1Y"]

        pwm_data = [joy1X - 100, joy1Y - 100]

        print(pwm_data)

        if use_serial:
            ser.write(((str(pwm_data[0]) + " " + str(pwm_data[1])) + str('\n')).encode())

        return 'OK', 200


@app.route('/FPV')
def control_FPV():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking
    running_video = True
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_ssd = False
    running_FPV = True
    running_tracking = False
    return render_template("page_fpv_from_face.html", disabled=disabled_button)


@app.route('/track')
def tracking():
    global running_video, running_color, running_size, \
        running_calibration, running_shape, running_white, \
        running_roundness, running_detect_obj, running_aruco, \
        running_characteristic, running_detect_line, running_exp, running_ssd, running_FPV, running_tracking
    running_video = False
    running_color = False
    running_size = False
    running_calibration = False
    running_shape = False
    running_white = False
    running_roundness = False
    running_detect_obj = False
    running_aruco = False
    running_characteristic = False
    running_detect_line = False
    running_exp = False
    running_ssd = False
    running_ssd = False
    running_FPV = False
    running_tracking = True
    return render_template("page_tracking.html", disabled=disabled_button)


def gen_white_balance(camera):
    global running_white, white_balance
    while running_white:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_size_detect(camera):
    global running_size, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA
    while running_size:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = size_detect(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_color_detect(camera):
    global running_color, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA
    while running_color:
        ret, frame = camera.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = color_detect(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_calibration(camera):
    global running_calibration
    while running_calibration:
        ret, image = camera.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = calibrationCam(image, FPS, width, height)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_shape_detect(camera):
    global running_shape, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA
    while running_shape:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = shape_call(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_roundness(camera):
    global running_roundness, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA
    while running_roundness:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = round_contr(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_detect_distance_between_obj(camera):
    global running_detect_obj, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA
    while running_detect_obj:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = detect_distance_between_obj(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen(camera):
    global running_video, camera_available
    while running_video:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_aruco(camera):
    global running_aruco
    while running_aruco:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = detect(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_charac(camera):
    global running_characteristic
    while running_characteristic:
        ret, image = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(image, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = character(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_line_detect(camera):
    global running_detect_line, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, LIMUN
    while running_detect_line:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame = lineDet(frame, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA, LIMUN)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_exposition(camera):
    global running_exp, blur, sharp, noise
    while running_exp:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_ssd(camera):
    global running_ssd
    while running_ssd:
        ret, frame = camera.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = ssd(frame, 0.55, 0.3)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def gen_tracking(camera):
    global running_tracking, pwm_data_track, use_serial, ser
    while running_tracking:
        ret, frame = camera.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = white_balancing(frame, white_balance)
        frame = exp(frame, blur, sharp, noise)
        frame, pwm_data_track = track_ob(frame, pwm_data_track)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        if use_serial:
            ser.write((str(pwm_data_track[0]) + " " + str(pwm_data_track[1]) + str('\n')).encode())
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_white_balance')
def video_feed_white_balance():
    return Response(gen_white_balance(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_color_detect')
def video_feed_color_detect():
    return Response(gen_color_detect(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_size_detect')
def video_feed_size_detect():
    return Response(gen_size_detect(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_caliration')
def video_feed_calibration():
    return Response(gen_calibration(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_shape_detect')
def video_feed_shape_detect():
    return Response(gen_shape_detect(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_roundness_detect')
def video_feed_roundness_detect():
    return Response(gen_roundness(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_detect_distance_between_obj')
def video_feed_detect_distance_between_obj():
    return Response(gen_detect_distance_between_obj(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_aruco_detect')
def video_feed_aruco_detect():
    return Response(gen_aruco(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_charac')
def video_feed_charac():
    return Response(gen_charac(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_line_detect')
def video_feed_line_detect():
    return Response(gen_line_detect(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_exp')
def video_feed_exp():
    return Response(gen_exposition(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_ssd')
def video_feed_ssd():
    return Response(gen_ssd(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_tracking')
def video_feed_tracking():
    return Response(gen_tracking(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)

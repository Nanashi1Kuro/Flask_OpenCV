import cv2
import numpy as np
from size_detect import size_detect, midpoint
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial import distance as dist
from color_detect import color_detect
import pandas as pd
import argparse

import locale
locale.setlocale(locale.LC_ALL, '')




# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

def get_area(c,frame):
    pixelsPerMetric = None
    # if the contour is not sufficiently large, ignore it
    if cv2.contourArea(c) < 200 or cv2.contourArea(c) > 40000:
        return frame
    # compute the rotated bounding box of the contour
    # orig = image.copy() <- moved outside FOR and outside IF len(cnts)
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    #cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    # draw the midpoints on the image
    cv2.circle(frame, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(frame, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # draw lines between the midpoints
    cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 360  # args["width"] SET TO RIGHT CAMERA VALUE PLEASE
    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    # draw the object sizes on the image
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




def shape_call(frame,HMIN,SMIN,VMIN,HMAX,SMAX,VMAX, AREA):


    lower_range = np.array([HMIN,SMIN,VMIN])
    upper_range = np.array([HMAX,SMAX,VMAX])
    hsv = frame # cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)
    #threshold = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    used_contours = []
    font = cv2.FONT_HERSHEY_COMPLEX
    """
    lower = {'red': ([136, 87, 111]), 'green': ([25, 52, 72]),
             'blue': ([94, 80, 2])}  # assign new item lower['blue'] = (93, 10, 0)
    upper = {'red': ([180, 255, 255]), 'green': ([177, 255, 255]), 'blue': ([120, 255, 255])}

    blurred = cv2.GaussianBlur(hsv, (11, 11), 0)
    hsv1 = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    """
    mlist = []
    clist = []
    ks = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 200: #or cv2.contourArea(cnt) > AREA:
            continue
        coloring = []

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
            cv2.putText(frame, coloring + " Треугольник", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

        elif len(approx) == 4:
            x2 = approx.ravel()[2]
            y2 = approx.ravel()[3]
            x4 = approx.ravel()[6]
            y4 = approx.ravel()[7]
            side1 = abs(x2 - x)
            side2 = abs(y4 - y)

            if abs(side1 - side2) <= 1:
                cv2.putText(frame, coloring + " Квадрат", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
            else:
                cv2.putText(frame, coloring + " Прямоугольник", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

        # elif 6 < len(approx) < 15:
        # cv2.putText(res, "Ellipse", (x, y), font, 1, (255,255,255))

        elif len(approx) > 10:
            cv2.putText(frame, coloring + " Круг", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    return frame




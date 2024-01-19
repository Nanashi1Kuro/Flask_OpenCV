from typing import Union, Tuple, Any, Optional

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def detect_distance_between_obj(image, HMIN, SMIN, VMIN, HMAX, SMAX, VMAX, AREA):
    lower_range = np.array([HMIN, SMIN, VMIN])
    upper_range = np.array([HMAX, SMAX, VMAX])
    hsv = image  # cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(mask, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        # sort the contours from left-to-right and, then initialize the
        # distance colors and reference object
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
                  (255, 0, 255))
        refObj = None

        # loop over the contours individually
        for c in cnts:
            area = cv2.contourArea(c)
            if (area < AREA) and (area > 600):
                cv2.drawContours(image, c, -1, (0), 5)
                if cv2.contourArea(c) < 100:
                    continue

                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)

                # compute the center of the bounding box
                cX = np.average(box[:, 0])
                cY = np.average(box[:, 1])

                # if this is the first contour we are examining (i.e.,
                # the left-most contour), we presume this is the
                # reference object
                if refObj is None:
                    # unpack the ordered bounding box, then compute the
                    # midpoint between the top-left and top-right points,
                    # followed by the midpoint between the top-right and
                    # bottom-right
                    (tl, tr, br, bl) = box
                    (tlblX, tlblY) = midpoint(tl, bl)
                    (trbrX, trbrY) = midpoint(tr, br)

                    # compute the Euclidean distance between the midpoints,
                    # then construct the reference object
                    D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    refObj = (box, (cX, cY), D / 1920)
                    continue

                # draw the contours on the image
                cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
                cv2.drawContours(image, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

                # stack the reference coordinates and the object coordinates
                # to include the object center
                refCoords = np.vstack([refObj[0], refObj[1]])
                objCoords = np.vstack([box, (cX, cY)])

                # loop over the original points
                # for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
                # draw circles corresponding to the current points and
                # connect them with a line
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


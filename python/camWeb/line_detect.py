import math
import cv2
import numpy as np
from scipy.spatial import distance as dist

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
    vX = bX-aX
    vY = bY-aY
    if(vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)

def lineDet(img,HMIN,SMIN,VMIN,HMAX,SMAX,VMAX, AREA, LIMUN):
    img = val_contrast(img, LIMUN)
    lower_range = np.array([HMIN,SMIN,VMIN])
    upper_range = np.array([HMAX,SMAX,VMAX])
    hsv = img
    gray = cv2.inRange(hsv, lower_range, upper_range)
    edges = cv2.Canny(gray, 0, 500, 5)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    try:
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
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
            if (area < AREA) and (area > 1000):
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
                    cv2.line(img, (int(xA), int(yA)), (int(xB), int(yB)), (0,255,0), 2)
                    cv2.putText(img, "{:.1f}".format(D), (int(mX), int(mY - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    except Exception as e:
        print(e)

    return img




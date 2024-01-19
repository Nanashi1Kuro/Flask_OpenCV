
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
    # converting to LAB color space
    if LIMUN == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(LIMUN, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

def lineDet(img,HMIN,SMIN,VMIN,HMAX,SMAX,VMAX, AREA, LIMUN):
    #img = cv2.imread("shapes.jpg")
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = val_contrast(img, LIMUN)
    lower_range = np.array([HMIN,SMIN,VMIN])
    upper_range = np.array([HMAX,SMAX,VMAX])
    hsv = img #cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# Filter the image and get the binary mask, where white represents
	# your target color
    gray = cv2.inRange(hsv, lower_range, upper_range)
	
    # Apply edge detection method on the image
    edges = cv2.Canny(gray, 0, 500, 5)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    try:
        # finding the contours

        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a*r

            # y0 stores the value rsin(theta)
            y0 = b*r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

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
                    xB = 300
                    yB = 300
                    cv2.putText(img, str(radius), center, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.drawContours(img, [cnt], 0, (0, 0, 0), 2)
                    D = (dist.euclidean((xA, yA), (xB, yB)) / 300)
                    (mX, mY) = midpoint((xA, yA), (xB, yB))
                    cv2.putText(img, "{:.1f}".format(D), (int(mX), int(mY - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    except Exception as e:
        pass

    return img





import cv2
import numpy as np


def lineDet(img,HMIN,SMIN,VMIN,HMAX,SMAX,VMAX, AREA):
    img = cv2.imread("shapes.jpg")
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:

            # take the first contour
            count = cnt
            area = cv2.contourArea(count)
            if (area < 100000) and (area > 200):
                M = cv2.moments(count)
                ((x_axis, y_axis), radius) = cv2.minEnclosingCircle(count)

                center = (int(x_axis), int(y_axis))
                radius = int(radius)
                cv2.putText(img, radius, (150, 150), cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 255, 0))
                cv2.drawContours(img, cnt, -1, (255, 255, 0), 2)

    except Exception as e:
        pass

    return img


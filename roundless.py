import numpy as np
import cv2

def round_contr(image,HMIN,SMIN,VMIN,HMAX,SMAX,VMAX, AREA):

    hsvFrame = image #cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([HMIN,SMIN,VMIN])
    upper_range = np.array([HMAX,SMAX,VMAX])

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv2.inRange(hsvFrame, lower_range, upper_range)

    # threshold = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    font = cv2.FONT_HERSHEY_COMPLEX


    for cnt in contours:
        if cv2.contourArea(cnt) < 200: #or cv2.contourArea(cnt) > 40000:
            continue
        #count = contours[0]
        (x_axis, y_axis), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x_axis), int(y_axis))
        radius = int(radius)
        image = cv2.putText(image, str(radius), center, font, 1,
                            (0, 0, 0), 2, cv2.LINE_AA, False)
        image = cv2.circle(image, center, radius, (0, 255, 0), 2)
    return image

from flask import Flask, render_template, Response
import numpy as np
import cv2

def color_detect(frame,HMIN,SMIN,VMIN,HMAX,SMAX,VMAX, AREA):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_range = np.array([HMIN,SMIN,VMIN])
    upper_range = np.array([HMAX,SMAX,VMAX])
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
        if area < 200 or area > AREA:
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
        if area < 200 or area > AREA:
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
        if area < 200 or area > AREA:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (255, 0, 0), 2)

            cv2.putText(frame, "Синий", (x, y),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.0, (255, 0, 0))
    return frame

import cv2
import numpy as np

def blur(image, b):
    # Load image as grayscale
    if b == 0:
        return image
    else:
        blurred = cv2.blur(image, (b, b))
        return blurred
def noise(image, n):
    if n == 0:
        return image
    else:
        gn_img = cv2.GaussianBlur(image, (15,15), n, n, cv2.BORDER_CONSTANT)
        return gn_img

def sharp(image, s):
    if s == 0:
        return image
    else:
        sharp_image = cv2.filter2D(image, -1, (s, s))
        return sharp_image
def exp(image, b, s, n):
    bluring = blur(image, b)
    noiseness = noise(bluring, n)
    sharping = sharp(noiseness, s)
    result = sharping
    return result


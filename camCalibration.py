import cv2
import numpy as np
import pickle

def calibrationCam(image, FPS, width, height):

    CHECKERBOARD = (9, 6)
    MIN_POINTS = 50
    RECORD = True
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Vector for the 3D points:
    threedpoints = []
    # Vector for 2D points:
    twodpoints = []
    # 3D points real world coordinates:
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    if RECORD:
        writer = cv2.VideoWriter('calibration.mp4', cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width, height))

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
            # When we have minimum number of data points, stop:
        if len(twodpoints) > MIN_POINTS:
            if RECORD: writer.release()
        image = cv2.drawChessboardCorners(image,CHECKERBOARD,corners2, ret)
        if RECORD:
            writer.write(image)
    if len(threedpoints) > 0:
        h, w = image.shape[:2]
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None, None)
        print(" Camera matrix:")
        print(matrix)

        print("\n Distortion coefficient:")
        print(distortion)

        print("\n Rotation Vectors:")
        print(r_vecs)

        print("\n Translation Vectors:")
        print(t_vecs)
        # transform the matrix and distortion coefficients to writable lists
        data = {'camera_matrix': np.asarray(matrix).tolist(),
                'dist_coeff': np.asarray(distortion).tolist()}

        # and save it to a file
        """
        f = open('calibration.pckl', 'wb')
        pickle.dump((matrix, distortion, r_vecs, t_vecs ), f)
        f.close()
        """

        cv_file = cv2.FileStorage("calibration_matrix.yaml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("camera_matrix", matrix)
        cv_file.write("dist_coeff", distortion)

    return image

from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, abort
import subprocess
from shlex import quote
import json
import numpy as np
import cv2
import time
import serial
import socket
from color_detect import color_detect
from size_detect import size_detect
from camCalibration import calibrationCam
from shape_detection_gray import shape_call
from white_balance_detect import white_balancing
from roundless import round_contr
from location_object import detect_distance_between_obj
from aruco_detect import detect
from charctiristic import character, check_CPU_temp, check_RAM_usage, check_CPU_usage
from line_detect import lineDet
from exposition import exp
from ssd_mobilenet import ssd
from For_serial import _map
from track_with_line import track_ob

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


pwm_data = [0, 0, 0, 0]
pwm_data_track = [0, 0]


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
AREA = 500

use_serial = True
running_video = False
camera_available = True

try:
    ser = serial.Serial("/dev/ttyS0", 9600, timeout=10)
    ser.flush()
except OSError as e:
    print("Низкое питание на плате!!!")
    use_serial = False


if not camera.isOpened():
    print("Камера не запущена")
    camera_available = False


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
    address = "https://192.168.8.1:6175"
    return render_template('page_chars_of_camera.html', cpu_temp=cpu_temp, cpu_usage=cpu_usage, ram_usage=ram_usage,
                           address=address, disabled=disabled_button)


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
            ser.write((str(pwm_data[0]) + " " + str(pwm_data[1])) + str('\n')).encode()

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
        # frame = white_balancing(frame, white_balance)
        # frame = exp(frame, blur, sharp, noise)
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
    app.run()
    #app.run(host='192.168.8.1', port=5050, debug=False, threaded=True)

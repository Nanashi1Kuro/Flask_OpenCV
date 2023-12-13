import numpy as np
import cv2
import time
import psutil
import subprocess
import re

start_time = time.time()
display_time = 2
fc = 0
FPS = 0

def character(image):
    global start_time, display_time, fc, FPS
    fc +=1
    TIME = time.time() - start_time
    if(TIME)>=display_time:
        FPS = fc/(TIME)
        fc = 0
        start_time = time.time()
    fps_disp = "FPS: " + str(FPS)[0:5]
    image = cv2.putText(image, fps_disp, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)
    return image

def check_CPU_temp():
    temp = None
    err, msg = subprocess.getstatusoutput('cat /sys/class/thermal/thermal_zone0/temp')
    if not err:
        m = re.search(r'-?\d\.?\d*', msg)  # a solution with a  regex
        try:
            temp = float(m.group())/1000
        except ValueError:  # catch only error needed
            pass
    return temp

def check_RAM_usage():
    return round(psutil.virtual_memory()[3]/1000000,2)
    
def check_CPU_usage():
    return psutil.cpu_percent(4)



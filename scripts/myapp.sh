#!/bin/bash
#while ! ping -c 1 -n -w 1 192.168.8.1 &> /dev/null
#do
#    printf "%c" "."
#done
cd /code/python/camWeb
export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.8/site-packages/"
python3 /code/python/camWeb/app.cpython-38.pyc >> /code/logs/myapp.log 2>&1

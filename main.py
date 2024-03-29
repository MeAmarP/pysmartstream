import argparse
import platform

import cv2

from flask import Flask, render_template, Response

from camera import VideoCamera, Color
from objectdetector import ObjectDetector


# TODO: Single Loggerclass to debug them all.
# TODO: Replace FlaskAPI by FastAPI


parser = argparse.ArgumentParser()
parser.add_argument("--path_to_model_config", help="Provide Path to DNN model config file.")
parser.add_argument("--path_to_model_weights", help="Provide Path to DNN model weights file.")
args = parser.parse_args()

app = Flask(__name__)

if platform.machine() == 'aarch64':
    src = 'v4l2src ! video/x-raw,width={},height={} ! videoconvert ! appsink'.format(640, 480)
else:
    src = 0
# Init video stream with source <Local Camera = 0 | Video file>, frame Width & Height
camera = VideoCamera(src=src, width=640, height=480)

ssd_mdl = ObjectDetector(args.path_to_model_weights, args.path_to_model_config)


def process_frames(cam):
    while True:
        ret, myframe = cam.video.read()
        objclass_values, objscore_values, objbboxes_values = ssd_mdl.detect_objects_in_frame(myframe)
        for objclass_ele, score_ele,bbox_ele in zip(objclass_values, objscore_values, objbboxes_values):
            cv2.rectangle(myframe,
                          (int(bbox_ele[0]), int(bbox_ele[1])),(int(bbox_ele[2] + bbox_ele[0]), int(bbox_ele[3] + bbox_ele[1])),
                          Color.yellow,
                          thickness=2)
        cam.draw_str(myframe,(20, 700),"Hello-World")
        frame = cam.encode_frame(myframe)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =============================================================================
# App Functions
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(process_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    host_ip = "127.0.0.1"
    # host_ip = "0.0.0.0"
    app.run(host=host_ip, debug=True, port=8888)

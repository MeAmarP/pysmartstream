import argparse
import platform

import cv2

from flask import Flask, render_template, Response

from camera import VideoCamera, Color
from detector import ObjectDetector


# TODO: Single Loggerclass to debug them all.
# TODO: Replace FlaskAPI by FastAPI


parser = argparse.ArgumentParser()
parser.add_argument("--video_file", help="Provide Path to Video file.")
parser.add_argument("--model_config", help="Provide Path to DNN model config file.")
parser.add_argument("--model_weights", help="Provide Path to DNN model weights file.")

args = parser.parse_args()

app = Flask(__name__)

camera = cv2.VideoCapture(args.video_file)  # Video File Source
yolo_detector = ObjectDetector(model_cfg=args.model_config, 
                     model_weights= args.model_weights,
                     class_file="./assets/coco.names")


def process_frames(cam):
    while True:
        ret, myframe = cam.read()
        # Check if the frame is valid
        if not ret or myframe is None:
            print("Error: Please provide a valid video file.")
            break
        detections = yolo_detector.detect(myframe, target_class_id=0) # 0 for person

        if len(detections) > 0:
            yolo_detector.display_detections(frame=myframe, detections=detections)

        _, frame_jpeg = cv2.imencode('.jpg', myframe)
        frame_jpeg = frame_jpeg.tobytes()
        # frame = cam.encode_frame(myframe)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_jpeg + b'\r\n')

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
#/home/c3po/Documents/project/learning/amar-works/yolo-triton-onnx/models/yolov3.cfg
#/home/c3po/Documents/project/learning/amar-works/yolo-triton-onnx/models/yolov3.weights
    host_ip = "127.0.0.1"
    # host_ip = "0.0.0.0"
    app.run(host=host_ip, debug=True, port=8888)

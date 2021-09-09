import cv2
from flask import Flask, render_template, Response
from camera import VideoCamera

# TODO: Stream using multi-threading so to have close-real-time feed.
# TODO: Single Loggerclass to debug them all.



app = Flask(__name__)

if platform.machine() == 'aarch64':
    src = 'v4l2src ! video/x-raw,width={},height={} ! videoconvert ! appsink'.format(640, 480)
else:
    src = 0
# Init video stream with source <Local Camera = 0 | Video file>, frame Width & Height
camera = VideoCamera(src=src, width=1280, height=720)

# path_to_model_config = r'C:\\Users\\H402321\\Documents\\projects\\year2020\\tev\\pretrained-models\\tflow-models\\object detection\\ssd_mobilenet_v1_coco_2018_01_28\\ssdmbnetv1coco.pbtxt'
# path_to_model_binfile = r'C:\\Users\\H402321\\Documents\\projects\\year2020\\tev\\pretrained-models\\tflow-models\\object detection\\ssd_mobilenet_v1_coco_2018_01_28\\frozen_inference_graph.pb'
ssd_mdl = ObjectDetector(args.path_to_model_weights, args.path_to_model_config)


def process_frames(cam):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    host_ip = "127.0.0.1"
    # host_ip = "0.0.0.0"
    app.run(host=host_ip, debug=True, port=8888)

import cv2
from flask import Flask, render_template, Response
from camera import VideoCamera

# TODO: Stream using multi-threading so to have close-real-time feed.
# TODO: Single Loggerclass to debug them all.



app = Flask(__name__)

def gen_frames(camera):
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

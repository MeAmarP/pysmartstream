"""
Camera.py

Objective:
- Read Camera stream from local device or Video file

TODO Add class method to change or transform camera props.
TODO Add method to compute camera performance like FPS, Latency etc.
TODO Add drawing methods(str, rect, circle) to annotate frames.
"""

import cv2
import platform

if platform.machine() == 'aarch64':
    src = 'v4l2src ! video/x-raw,width={},height={} ! videoconvert ! appsink'.format(640, 480)
else:
    src = 0

class VideoCamera(object):
    def __init__(self,src=0, width=640, height=480):
        self.video = cv2.VideoCapture(src)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __del__(self):
        self.video.release()

    def __str__(self):
        return f"Created Camera Stream of W = {self.video.get(cv2.CAP_PROP_FRAME_WIDTH)}," \
               f"H = {self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)}"

    def encode_frame(self,frame):
        ret, frame_jpeg = cv2.imencode('.jpg', frame)
        return frame_jpeg.tobytes()

        # cv2.putText(image, (f"Detect Class-->{objclass_values}"), (10,430), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        # cv2.putText(image, (f"Detect Score-->{objscore_values}"), (10,450), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        # cv2.putText(image, (f"Detect BBox-->{objbboxes_values}"), (10,470), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
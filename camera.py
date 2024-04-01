"""
Camera.py

Objective:
- Read Camera stream from local device or Video file

TODO Add class method to change or transform camera props.
TODO Add method to compute camera performance like FPS, Latency etc.
TODO Add drawing methods(str, rect, circle) to annotate frames.
"""

import cv2
import traceback

class Color:
    """
    VIBGYOR+BW
    """
    violet = (211,0,148)
    indigo = (130, 0,75)
    blue = (255,0,0)
    green = (0,255,0)
    yellow = (0,255,255)
    orange = (0,127,255)
    red = (0,0,255)
    black = (0,0,0)
    white = (255,255,255)


class VideoCamera(object):
    def __init__(self,src=0, width=640, height=480):
        try:
            self.video = cv2.VideoCapture(src)
            if not self.video.isOpened():
                raise IOError("Failed to open webcam.")
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {str(e)}")
            exit()
        # self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    def __del__(self):
        self.video.release()

    def __str__(self):
        return f"Created Camera Stream of W = {self.video.get(cv2.CAP_PROP_FRAME_WIDTH)}," \
               f"H = {self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)}"

    def encode_frame(self,frame):
        ret, frame_jpeg = cv2.imencode('.jpg', frame)
        return frame_jpeg.tobytes()

    def draw_str(self,img,target,s):
        x,y = target
        cv2.putText(img,str(s),(x+1,y+1),cv2.FONT_HERSHEY_PLAIN,1,Color.green,thickness=2,lineType=cv2.LINE_AA)
        cv2.putText(img,str(s), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, Color.black, lineType=cv2.LINE_AA)


    def draw_rect(self):
        pass

    def draw_circle(self):
        pass


if __name__ == '__main__':
    try:
        cam = VideoCamera(src=0,width=1280,height=720)
        print(cam)
        ret,img = cam.video.read()
        print(img.shape)
    except Exception as e:
        traceback.print_exc()
        print("Error occurred while processing the camera feed", str(e))
    finally:
        if 'cam' in locals() or 'cam' in globals():  # checks if `cam` variable is defined
            cam.video.release()


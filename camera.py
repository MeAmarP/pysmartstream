import cv2
import platform

from objectdetector import ObjectDetector

if platform.machine() == 'aarch64':
    src = 'v4l2src ! video/x-raw,width={},height={} ! videoconvert ! appsink'.format(640, 480)
else:
    src = 0

path_to_model_config = 'pretrained-models\\tflow-models\\object detection\\ssd_mobilenet_v1_coco_2018_01_28\\ssdmbnetv1coco.pbtxt'
path_to_model_binfile = 'pretrained-models\\tflow-models\\object detection\\ssd_mobilenet_v1_coco_2018_01_28\\frozen_inference_graph.pb'
ssd_mdl = ObjectDetector(path_to_model_binfile, path_to_model_config)


class VideoCamera(object):
    def __init__(self,src=0, width=640, height=480):
        self.video = cv2.VideoCapture(src)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        objclass_values, objscore_values, objbboxes_values = ssd_mdl.detect_objects_in_frame(image)
        for objclass_ele, score_ele,bbox_ele in zip(objclass_values, objscore_values, objbboxes_values):
            cv2.rectangle(image, (int(bbox_ele[0]), int(bbox_ele[1])),(int(bbox_ele[2] + bbox_ele[0]), int(bbox_ele[3] + bbox_ele[1])),
                          (0, 255, 0),
                          thickness=2)

        # cv2.putText(image, (f"Detect Class-->{objclass_values}"), (10,430), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        # cv2.putText(image, (f"Detect Score-->{objscore_values}"), (10,450), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        # cv2.putText(image, (f"Detect BBox-->{objbboxes_values}"), (10,470), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
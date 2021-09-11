import cv2
import numpy as np

# TODO Add method to perform inference with tflite models

class ObjectDetector:
    def __init__(self,path_to_model_binfile, path_to_model_config_file,dnn_input_size = 300):
        self.net = cv2.dnn.readNet(path_to_model_binfile, path_to_model_config_file)
        inp = np.random.standard_normal([1, 3, dnn_input_size, dnn_input_size]).astype(np.float32)
        self.net.setInput(inp)

    def __str__(self):
        pass

    def detect_objects_in_frame(self,frame):
        net = cv2.dnn_DetectionModel(self.net)
        objclass, objscore, objbboxes = net.detect(frame)
        return objclass, objscore, objbboxes

if __name__ == '__main__':
    path_to_model_config = ''
    path_to_model_binfile = ''
    path_to_input_img = r""

    #init model
    ssd_mdl = ObjectDetector(path_to_model_binfile, path_to_model_config)

    testimg = cv2.imread(path_to_input_img)
    objclass_values, objscore_values, objbboxes_values = ssd_mdl.detect_objects_in_frame(testimg)
    print(f"Detected Classes -->{objclass_values.item()} \nDetected Score -->{objscore_values.item()} \nDetected BBox -->{objbboxes_values.tolist()}")







import cv2
import numpy as np

# TODO: Add Class to load & run classification model
# TODO: Add Class to load & run keypoints model
# TODO: Add Class to load & run segmentation model

class ObjectDetector:
    """
    Based on OpenCV - DetectionModel
    For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.

    More Docs <https://docs.opencv.org/4.5.0/d3/df1/classcv_1_1dnn_1_1DetectionModel.html#details>
    """
    # TODO Add init-method to load label files having Key-Value/dict of target classes and class_ids.
    def __init__(self,path_to_model_binfile, path_to_model_config_file,dnn_input_size = 300):
        self.net = cv2.dnn.readNet(path_to_model_binfile, path_to_model_config_file)
        inp = np.random.standard_normal([1, 3, dnn_input_size, dnn_input_size]).astype(np.float32)
        self.net.setInput(inp)

    def __str__(self):
        return f"Loaded Model-type ---> {ObjectDetector.__name__}"

    def detect_objects_in_frame(self,frame):
        net = cv2.dnn_DetectionModel(self.net)
        objclass, objscore, objbboxes = net.detect(frame)
        return objclass, objscore, objbboxes

class ObjectClassifier:
    """
    Note: Tested with caffe and tflow DNN models.
    """
    def __init__(self,path_model_weights, path_model_config,dnn_input_size = 224):
        self.net = cv2.dnn.readNet(path_model_weights, path_model_config)
        self.dnn_input_size = dnn_input_size
        inp = np.random.standard_normal([1, 3, dnn_input_size, dnn_input_size]).astype(np.float32)
        self.net.setInput(inp)

    def __str__(self):
        return f"Loaded Model-type ---> {ObjectClassifier.__name__}"

    def classify_objects_in_frame(self,frame):
        net = cv2.dnn_ClassificationModel(self.net)
        net.setInputParams(size=(self.dnn_input_size,self.dnn_input_size),
                           mean = (0.485, 0.456, 0.406),
                           scale = 0.003921,
                           swapRB=True,
                           crop=False,)
        cls_id, conf_score = net.classify(frame)
        return cls_id, conf_score


if __name__ == '__main__':
    path_to_model_config = ''
    path_to_model_binfile = 'models/mobilenetv2_onnx/mobilenetv2-7.onnx'


    # init model
    # ssd_mdl = ObjectDetector(path_to_model_binfile, path_to_model_config)

    classify_mdl = ObjectClassifier(path_to_model_binfile, path_to_model_config,dnn_input_size=224)
    img = cv2.imread('sample_imgs/tiget.jpg')
    print(classify_mdl.classify_objects_in_frame(img))
    # LayersIds, inLayersShapes, outLayersShapes = classify_mdl.net.getLayersShapes([1,3,224,224])
    # print(f"LayerIDS{LayersIds}\nINputLayerShapes{inLayersShapes}\nOutputLayerShapes{outLayersShapes}")

# pysmartstream (VideoInsight)
## About Project
**VideoInsight** is a cutting-edge web application designed to provide real-time analytics on video files streamed through a user-friendly web interface. Leveraging the power of OpenCV for video processing and Flask for backend development, VideoInsight offers a seamless experience for users looking to extract valuable insights from their video data. From motion detection to advanced object recognition, VideoInsight caters to a wide range of analytical needs, making it an indispensable tool for security, research, and content analysis

## Features:

- **Video Upload and Streaming:** Users can easily upload video files to the platform. The application supports streaming of uploaded videos directly within the web interface, facilitating immediate analysis without the need for downloads or external players.

- **Real-Time Analytics:** Powered by OpenCV, VideoInsight performs real-time analysis on streamed videos. The application can detect and track objects, analyze motion patterns, and provide customizable analytics based on user-defined criteria.

- **Interactive Dashboard:** The heart of VideoInsight is its interactive dashboard, where users can view real-time analytics results overlaid on the streamed video. The dashboard includes tools for filtering results, adjusting analysis parameters, and navigating through video frames.

## Usage
- Download yolov3/4 model weights and config files from the official YOLO website or any other source. 
- Download coco.names file. 
- Place these files in the root directory of your project.
- Run main.py with path to and weights as arguments.

Examples:
```bash
python main.py --path_to_model_config yolov3.cfg --path_to_model_weights yolov3.weights

python main.py --path_to_model_config ssd_mobilenet_v2_coco.config --path_to_model_weights ssd_mobilenet_v2_coco_checkpoint
```
### Options:

- `-h, --help`: Show help message

- `--path_to_model_config PATH_TO_MODEL_CONFIG`: Provide Path to DNN model config file.

- `--path_to_model_weights PATH_TO_MODEL_WEIGHTS`: Provide Path to DNN model weights file.

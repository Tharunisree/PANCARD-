# import cv2
# import os
# import time
# import uuid
# import subprocess
# import tensorflow as tf
# from object_detection.utils import config_util
# from object_detection.protos import pipeline_pb2
# from google.protobuf import text_format
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder

# # Directory Paths
# WORKSPACE_PATH = 'Tensorflow/workspace'
# SCRIPTS_PATH = 'Tensorflow/scripts'
# APIMODEL_PATH = 'Tensorflow/models'
# ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
# IMAGE_PATH = os.path.join(WORKSPACE_PATH, 'images')
# MODEL_PATH = os.path.join(WORKSPACE_PATH, 'models')
# PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'pre-trained-models')
# CONFIG_PATH = os.path.join(MODEL_PATH, 'my_ssd_mobnet', 'pipeline.config')
# CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'my_ssd_mobnet')
# os.makedirs(ANNOTATION_PATH, exist_ok=True)


# # Label Map
# labels = [{'name': 'signature', 'id': 1}]
# label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')

# # Open the label map file for writing
# with open(label_map_path, 'w') as f:
#     for label in labels:
#         f.write('item {\n')
#         f.write(f'  name: "{label["name"]}"\n')
#         f.write(f'  id: {label["id"]}\n')
#         f.write('}\n')

# # TF Record Generation
# def generate_tf_record(image_dir, label_map_path, output_record_path):
#     command = f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {image_dir} -l {label_map_path} -o {output_record_path}"
#     subprocess.run(command, shell=True, check=True)

# generate_tf_record(os.path.join(IMAGE_PATH, 'train'), label_map_path, os.path.join(ANNOTATION_PATH, 'train.record'))
# generate_tf_record(os.path.join(IMAGE_PATH, 'test'), label_map_path, os.path.join(ANNOTATION_PATH, 'test.record'))


# # 3. Download TF Models Pretrained Models from Tensorflow Model Zoo

# if not os.path.exists(os.path.join(PRETRAINED_MODEL_PATH, 'models')):
#     os.system(f"git clone https://github.com/tensorflow/models {PRETRAINED_MODEL_PATH}/models")

# # 4. Copy Model Config to Training Folder
# CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

# if not os.path.exists(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)):
#     os.mkdir(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME))

# os.system(f"copy {os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config')} {os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)}")

# # 5. Update Config For Transfer Learning
# CONFIG_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'pipeline.config')

# config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

# with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
#     proto_str = f.read()
#     text_format.Merge(proto_str, pipeline_config)

# len(labels)
# pipeline_config.model.ssd.num_classes = 2
# pipeline_config.train_config.batch_size = 4
# pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'checkpoint', 'ckpt-0')
# pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
# pipeline_config.train_input_reader.label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
# pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'train.record')]
# pipeline_config.eval_input_reader[0].label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
# pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'test.record')]
# config_text = text_format.MessageToString(pipeline_config)

# with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
#     f.write(config_text)

# # 6. Train the model
# train_command = f"""python {APIMODEL_PATH}/research/object_detection/model_main_tf2.py --model_dir={MODEL_PATH}/{CUSTOM_MODEL_NAME} --pipeline_config_path={MODEL_PATH}/{CUSTOM_MODEL_NAME}/pipeline.config --num_train_steps=5000"""

# print(train_command)

# # 7. Load Train Model From Checkpoint
# # Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

# # @tf.function
# # def detect_fn(image):



# @tf.function
# # def detect_fn(image):
# #     desired_size = (300, 300)  # Adjust the dimensions as needed

# #     # Convert the TensorFlow tensor to a NumPy array within a TensorFlow function
# #     image = tf.image.convert_image_dtype(image, tf.uint8)

# #     # Resize the input image using OpenCV
# #     image = tf.image.resize(image, desired_size)
    
# #     # Convert the resized image back to a TensorFlow tensor
# #     image = tf.convert_to_tensor(image, dtype=tf.uint8)
# #     image = tf.expand_dims(image, 0)

# #     image, shapes = detection_model.preprocess(image)
# #     prediction_dict = detection_model.predict(image, shapes)
# #     detections = detection_model.postprocess(prediction_dict, shapes)

# #     # Encode the image as JPEG
# #     encoded_image = tf.image.encode_jpeg(image[0], format='grayscale')

# #     return detections, encoded_image

# @tf.function
# def detect_fn(image):
#     desired_size = (300, 300)  # Adjust the dimensions as needed

#     # Convert the TensorFlow tensor to a NumPy array within a TensorFlow function
#     image = tf.image.convert_image_dtype(image, tf.uint8)

#     # Resize the input image using OpenCV
#     image = tf.image.resize(image, desired_size)

#     # Convert the resized image back to a TensorFlow tensor
#     image = tf.convert_to_tensor(image, dtype=tf.uint8)
#     image = tf.expand_dims(image, 0)

#     # Rest of your detection code
#     image, shapes = detection_model.preprocess(image)
#     prediction_dict = detection_model.predict(image, shapes)
#     detections = detection_model.postprocess(prediction_dict, shapes)

#     # Encode the image as JPEG
#     encoded_image = tf.image.encode_jpeg(image[0], format='grayscale')

#     return detections, encoded_image


#     # image, shapes = detection_model.preprocess(image)
#     # prediction_dict = detection_model.predict(image, shapes)
#     # detections = detection_model.postprocess(prediction_dict, shapes)
#     # return detections

# import cv2
# import numpy as np
# from object_detection.utils import label_map_util
# def signature():
# # Initialize a counter for saved images
#     image_counter = 0

#     category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'label_map.pbtxt'))
#     cap = cv2.VideoCapture(0)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     while True:
#         ret, frame = cap.read()
#         image_np = np.array(frame)

#         input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
#         detections = detect_fn(input_tensor)

#         num_detections = int(detections.pop('num_detections'))
#         detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
#         detections['num_detections'] = num_detections

#         # detection_classes should be ints.
#         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#         label_id_offset = 1
#         image_np_with_detections = image_np.copy()

#         viz_utils.visualize_boxes_and_labels_on_image_array(
#             image_np_with_detections,
#             detections['detection_boxes'],
#             detections['detection_classes'] + label_id_offset,
#             detections['detection_scores'],
#             category_index,
#             use_normalized_coordinates=True,
#             max_boxes_to_draw=5,
#             min_score_thresh=.5,
#             agnostic_mode=False)

#         cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

#         # Check if a signature is detected
#         if 1 in detections['detection_classes']:
#             # Save the image with detections
#             image_counter += 1
#             image_name = f"detected_signature_{image_counter}.jpg"
#             cv2.imwrite(image_name, image_np_with_detections)
#             print(f"Signature detected and image {image_name} saved!")
            
#             # Release the camera after capturing the image
#             cap.release()

#         key = cv2.waitKey(1) & 0xFF
#         print(f"Key pressed: {key}")  # Debug statement

#         if key == ord('q'):
#             cap.release() 
#             cv2.destroyAllWindows()  # Make sure to close the OpenCV window
#             break
# if __name__ == "__main__":
#     signature()














import cv2
import os
import time
import uuid
import subprocess
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# Directory Paths
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, 'annotations')
IMAGE_PATH = os.path.join(WORKSPACE_PATH, 'images')
MODEL_PATH = os.path.join(WORKSPACE_PATH, 'models')
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, 'pre-trained-models')
CONFIG_PATH = os.path.join(MODEL_PATH, 'my_ssd_mobnet', 'pipeline.config')
CHECKPOINT_PATH = os.path.join(MODEL_PATH, 'my_ssd_mobnet')
os.makedirs(ANNOTATION_PATH, exist_ok=True)
# Label Map
labels = [{'name': 'signature', 'id': 1}]
label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')

# Open the label map file for writing
with open(label_map_path, 'w') as f:
    for label in labels:
        f.write('item {\n')
        f.write(f'  name: "{label["name"]}"\n')
        f.write(f'  id: {label["id"]}\n')
        f.write('}\n')

# TF Record Generation
def generate_tf_record(image_dir, label_map_path, output_record_path):
    command = f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {image_dir} -l {label_map_path} -o {output_record_path}"
    subprocess.run(command, shell=True, check=True)

generate_tf_record(os.path.join(IMAGE_PATH, 'train'), label_map_path, os.path.join(ANNOTATION_PATH, 'train.record'))
generate_tf_record(os.path.join(IMAGE_PATH, 'test'), label_map_path, os.path.join(ANNOTATION_PATH, 'test.record'))

# Rest of the code remains the same
# ...
# 3. Download TF Models Pretrained Models from Tensorflow Model Zoo


if not os.path.exists(os.path.join(PRETRAINED_MODEL_PATH, 'models')):
    os.system(f"git clone https://github.com/tensorflow/models {PRETRAINED_MODEL_PATH}/models")

# 4. Copy Model Config to Training Folder
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

if not os.path.exists(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)):
    os.mkdir(os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME))

os.system(f"copy {os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config')} {os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)}")

# 5. Update Config For Transfer Learning
CONFIG_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'pipeline.config')

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

len(labels)
pipeline_config.model.ssd.num_classes = 2
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, 'test.record')]
config_text = text_format.MessageToString(pipeline_config)

with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
    f.write(config_text)

# 6. Train the model
train_command = f"""python {APIMODEL_PATH}/research/object_detection/model_main_tf2.py --model_dir={MODEL_PATH}/{CUSTOM_MODEL_NAME} --pipeline_config_path={MODEL_PATH}/{CUSTOM_MODEL_NAME}/pipeline.config --num_train_steps=5000"""

print(train_command)

# 7. Load Train Model From Checkpoint
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# # 8. Detect in Real-Time
import cv2
import numpy as np
def signature():

# Import necessary libraries and variables here (e.g., ANNOTATION_PATH, label_map_util, detect_fn)

# Initialize the category index and capture from the webcam
    category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'label_map.pbtxt'))
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize a counter for saved images
    image_counter = 0

    while True:
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            # detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.5,
            # line_thickness=2,
            agnostic_mode=False)
        # Now display a fixed class name for all detections
        # class_name = "detected signature"
        # cv2.putText(
        #     image_np_with_detections,
        #     class_name,
        #     (10, 30),  # Adjust the position where the class name is displayed
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,  # Font scale
        #     (0, 255, 0),  # Font color (green)
        #     2)  # Font thickness


        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        # Check for the 's' key press to save the image
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the image with detections
            image_counter += 1
            image_name = f"detected_signature_{image_counter}.jpg"
            cv2.imwrite(image_name, image_np_with_detections)
            print(f"Image saved as {image_name}")

        # Check for the 'q' key press to exit the loop
        elif key == ord('q'):
            cap.release()
            break
if __name__ == "__main__":
    signature()





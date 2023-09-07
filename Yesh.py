import subprocess
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from db import SessionLocal, engine  # Import the database components from db.py
import cv2
import json
import models
import os
import db
import uuid
from Tutorial import detect_fn
from  models import Image
app = FastAPI()

# Define the paths and other constants
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

# Define the label map
labels = [{'name': 'signature', 'id': 1}]
label_map_path = os.path.join(ANNOTATION_PATH, 'label_map.pbtxt')

# Function to generate TFRecord
def generate_tf_record(image_dir, label_map_path, output_record_path):
    command = f"python {SCRIPTS_PATH}/generate_tfrecord.py -x {image_dir} -l {label_map_path} -o {output_record_path}"
    subprocess.run(command, shell=True, check=True)

# Generate TFRecord files if they don't exist
if not os.path.exists(os.path.join(ANNOTATION_PATH, 'train.record')):
    generate_tf_record(os.path.join(IMAGE_PATH, 'train'), label_map_path, os.path.join(ANNOTATION_PATH, 'train.record'))

if not os.path.exists(os.path.join(ANNOTATION_PATH, 'test.record')):
    generate_tf_record(os.path.join(IMAGE_PATH, 'test'), label_map_path, os.path.join(ANNOTATION_PATH, 'test.record'))

category_index = label_map_util.create_category_index_from_labelmap(os.path.join(ANNOTATION_PATH, 'label_map.pbtxt'))
###################################
# # Define a route to receive a POST request to start signature detection
# @app.post("/detect_signature/")
# async def start_detection():
#     # Initialize the webcam capture
#     cap = cv2.VideoCapture(0)

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

#         # Check for the 's' key press to save the image
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('s'):
#             # Save the image with detections
#             image_name = f"detected_signature.jpg"
#             cv2.imwrite(image_name, image_np_with_detections)  # Save the annotated image

#             # Read the saved image as binary data
#             with open(image_name, "rb") as image_file:
#                 image_data = image_file.read()

#             # Insert the image into the database
#             with SessionLocal() as db_session:
#                 db_image = Image(data=image_data)  # Assuming 'data' is the binary data column
#                 db_session.add(db_image)
#                 db_session.commit()

#             print(f"Image saved and inserted into the database as {image_name}")

#         cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

#         # Check for the 'q' key press to exit the loop
#         if key == ord('q'):
#             cap.release()
#             break

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()





# Define a route to receive a POST request to start signature detection
# @app.post("/detect_signature/")
# async def start_detection(db:Session = Depends(get_db)):
#     # Initialize the webcam capture
#     cap = cv2.VideoCapture(0)

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

#         # Check for the 's' key press to save the image
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('s'):
#             # Save the image with detections
#             image_name = f"detected_signature.jpg"
#             cv2.imwrite(image_name, image_np_with_detections)
#             # Save the annotated image
#         try:
#             db_signature = Image(
#                 image=open('detected_signature.jpg', 'rb').read()
#             )
#             db.add(db_signature)
#             db.commit()
#         except Exception as e:
#             db.rollback()
#             # Log the error for debugging purposes
#             print(f"Error saving image to the database: {e}")
#             raise HTTPException(status_code=500, detail='Error saving image')



#             # # Read the saved image as binary data and resize it
#             # with open(image_name, "rb") as image_file:
#             #     image_data = image_file.read()
#             #     # Resize the image (adjust dimensions as needed)
#             #     resized_image = cv2.resize(cv2.imdecode(np.frombuffer(image_data, np.uint8), -1), (800, 600))
#             #     # Encode the resized image as binary data
#             #     # image_data = cv2.imencode('.jpg', resized_image)[1].tostring()
#             #     image_data = cv2.imencode('.jpg', resized_image)[1].tobytes()

#             # # Insert the image into the database
#             # with SessionLocal() as db_session:
#             #     db_image = Image(data=image_data)  # Assuming 'data' is the binary data column
#             #     db_session.add(db_image)
#             #     db_session.commit()

#             # print(f"Resized and saved image inserted into the database as {image_name}")

#         cv2.imshow('object detection', cv2.resize(image_np_with_detections, (400, 200)))

#         # Check for the 'q' key press to exit the loop
#         if key == ord('q'):
#             cap.release()
#             break

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)



# ... (previous code)
###############################################################################
# Define a route to receive a POST request to start signature detection
# @app.post("/detect_signature/")
# async def start_detection(db: Session = Depends(get_db)):
#     # Initialize the webcam capture
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         image_np = np.array(frame)

#         # Resize the captured image (adjust dimensions as needed)
#         resized_image = cv2.resize(image_np, (800, 600))

#         input_tensor = tf.convert_to_tensor(np.expand_dims(resized_image, 0), dtype=tf.float32)
#         detections = detect_fn(input_tensor)

#         num_detections = int(detections.pop('num_detections'))
#         detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
#         detections['num_detections'] = num_detections

#         # detection_classes should be ints.
#         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#         label_id_offset = 1
#         image_np_with_detections = resized_image.copy()

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

#         # Check for the 's' key press to save the image
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('s'):
#             # Save the image with detections
#             image_name = f"detected_signature.jpg"
#             cv2.imwrite(image_name, image_np_with_detections)
#             # Save the annotated image

#             try:
#                 db_signature = Image(
#                     image=open('detected_signature.jpg', 'rb').read()
#                 )
#                 db.add(db_signature)
#                 db.commit()
#             except Exception as e:
#                 db.rollback()
#                 # Log the error for debugging purposes
#                 print(f"Error saving image to the database: {e}")
#                 raise HTTPException(status_code=500, detail='Error saving image')

#         cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

#         # Check for the 'q' key press to exit the loop
#         if key == ord('q'):
#             cap.release()
#             break




@app.post("/detect_signature/")
async def start_detection(
        card_image: UploadFile,
        db: Session = Depends(get_db)
):
    try:
        # Save the uploaded card image to a temporary file
        with open("card_image.jpg", "wb") as temp_image:
            temp_image.write(card_image.file.read())

        # Load the saved image
        image_np = tf.image.decode_image(tf.io.read_file("card_image.jpg"))
         # Convert image data to float32
        image_np = tf.image.convert_image_dtype(image_np, tf.float32)


        # Expand dimensions if needed
        if len(image_np.shape) == 3:
            image_np = tf.expand_dims(image_np, axis=0)

        # Load the model and perform object detection
        input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1

        image_np_with_detections = image_np[0].numpy().copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.5,
            agnostic_mode=False
        )

        # Save the annotated image
        image_name = f"detected_signature.jpg"
        cv2.imwrite(image_name, image_np_with_detections)

        # Save the annotated image to the database
        try:
            db_signature = Image(
                image=open('detected_signature.jpg', 'rb').read()
            )
            db.add(db_signature)
            db.commit()
        except Exception as e:
            db.rollback()
            # Log the error for debugging purposes
            print(f"Error saving image to the database: {e}")
            raise HTTPException(status_code=500, detail='Error saving image')

        # Return the annotated image as a response
        return JSONResponse(content={"message": "Signature detected and saved successfully!"})

    except Exception as e:
        # Handle any exceptions that may occur
        return HTTPException(status_code=500, detail=f"An error occurred: {e}")


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="127.0.0.1", port=8000)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

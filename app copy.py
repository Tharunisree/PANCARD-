from models import ColorEntry1  # Import the new class
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from col_ext_medium import capture_frames, exact_color
from fastapi.responses import StreamingResponse,JSONResponse
from db import SessionLocal
from fastapi import FastAPI, Depends, status,HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import engine, SessionLocal
import models
import json
import cv2
import io
import json


from models import CapturedImage

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
# import dlib
import io
from PIL import Image
import PIL.Image
from models import CapturedImage
from db import SessionLocal, engine
import uvicorn
import numpy as np


app = FastAPI()
models.Base.metadata.create_all(bind=engine)

class UserBase(BaseModel):
    c_code: str
    percentage: float


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ColorEntry1Response(BaseModel):
    id: int
    image: bytes
    color_data: str




@app.post('/process')
async def process_card_region(db: Session = Depends(get_db)):
    # Call the capture_frames() function to capture the card region
    card_region = capture_frames()

    # Save the captured card region as card_image.jpg
    cv2.imwrite('card_image.jpg', card_region)

    color_data = exact_color('card_image.jpg', 900, 12, 2.5)
    
    # Convert color data to dictionary format {"color_code": "percentage"}
    color_dict = {entry['c_code']: str(round(entry['percentage'], 2)) for entry in color_data}
    
    # Save image and color data to the database (using ColorEntry1 class)
    try:
        db_color_entry = ColorEntry1(
            image=open('card_image.jpg', 'rb').read(),  # Store the image as BLOB data
            color_data=json.dumps(color_dict)  # Convert to JSON format
        )
        db.add(db_color_entry)
        db.commit()
        
        return {'color_data': color_dict}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail='Error saving image and color data')

# ...

import base64

# ...

from fastapi.responses import StreamingResponse, JSONResponse
import io
import json



############ both image and color data , the image in base-64 format

@app.get('/color_entries_1/{entry_id}', status_code=status.HTTP_200_OK)
async def get_color_entry(entry_id: int, db: Session = Depends(get_db)):
    db_entry = db.query(models.ColorEntry1).filter(models.ColorEntry1.id == entry_id).first()
    if db_entry is None:
        raise HTTPException(status_code=404, detail='Color entry not found')
    
    image_data = db_entry.image
    color_data = db_entry.color_data
    
    # Convert color data to a dictionary if it's in JSON format
    color_dict = json.loads(color_data)
    
    # Convert the image data to a Base64 encoded string
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Create a response JSON containing Base64 image and color data
    response_data = {
        "image_base64": image_base64,
        "color_data": color_dict
    }
    
    # Send the response JSON
    return JSONResponse(content=response_data, media_type="application/json")



######## it will print only image in postman 

@app.get('/color_entries_1/{entry_id}', status_code=status.HTTP_200_OK)
async def get_color_entry(entry_id: int, db: Session = Depends(get_db)):
    db_entry = db.query(models.ColorEntry1).filter(models.ColorEntry1.id == entry_id).first()
    if db_entry is None:
        raise HTTPException(status_code=404, detail='Color entry not found')
    
    image_data = db_entry.image
    color_data = db_entry.color_data
    
    return StreamingResponse(io.BytesIO(image_data), media_type='image/jpeg')




############################################################  KIRAN - FACE   ########################################################

from models import CapturedImage

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
# import dlib
import io
# from PIL import Image
import PIL.Image
from models import CapturedImage
from db import SessionLocal, engine
import uvicorn
import numpy as np
# from _dlib_pybind11 import *
# app = FastAPI()

# Initialize the database
CapturedImage.metadata.create_all(bind=engine)

# Initialize other variables
capture_completed = False  # Flag to indicate if the capture has been completed

# # Function to get a database session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

@app.post("/capture/")
async def capture_image(db: Session = Depends(get_db)):
    # Simulate image capture from a file
    image_path = 'card_image.jpg'
    img = PIL.Image.open(image_path)
    img_gray = img.convert('L')
 
    # Convert PIL image to OpenCV format
    img_cv2 = np.array(img_gray)

    # Initialize Haarcascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(img_cv2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Capture the detected face if it exists
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Add ROI margin
        roi_margin = 20
        x_roi, y_roi = max(0, x - roi_margin), max(0, y - roi_margin)
        w_roi, h_roi = w + 2 * roi_margin, h + 2 * roi_margin

        # Crop the detected face within the ROI
        detected_face_roi = img.crop((x_roi, y_roi, x_roi + w_roi, y_roi + h_roi))
        image_bytes = io.BytesIO()
        detected_face_roi.save(image_bytes, format="JPEG")

        # Store the image data in the database
        captured_image = CapturedImage(image_data=image_bytes.getvalue())
        db.add(captured_image)
        db.commit()

        return {"message": "Face Captured and Stored"}

    return {"message": "No Face Detected"}

# API route to get a stored image by ID
@app.get('/image/{image_id}')
async def get_image(image_id: int, db: Session = Depends(get_db)):
    db_entry = db.query(CapturedImage).filter(CapturedImage.id == image_id).first()
    if db_entry is None:
        raise HTTPException(status_code=404, detail='Image not found')

    return StreamingResponse(io.BytesIO(db_entry.image_data), media_type='image/jpeg')




# ###-------------------------------------- teja ------------------------------------


import os
import logging
import uvicorn
import numpy as np
# from PIL import Image
import PIL.Image
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from skimage import color, filters, measure
from db import SessionLocal
from models import ImageEntry1
from io import BytesIO
from FingerprintImageEnhancer import FingerprintImageEnhancer


def enhance_fingerprint(image):
    # Your fingerprint enhancement logic here
    # For example, histogram equalization
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = histogram.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    image_equalized = image_equalized.reshape(image.shape).astype(np.uint8)
    return image_equalized

@app.post("/process_fingerprint/")
async def process_fingerprint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        image = PIL.Image.open(file.file).convert("L")  # Convert to grayscale
        image_np = np.array(image)

        threshold = filters.threshold_otsu(image_np)
        binary_image = image_np < threshold

        labeled_image = measure.label(binary_image)
        region_properties = measure.regionprops(labeled_image)
        region_properties = sorted(region_properties, key=lambda r: r.area, reverse=True)

        thumbprint_region = None
        for region in region_properties:
            if region.major_axis_length > 100 and 0.5 < region.minor_axis_length / region.major_axis_length < 0.9:
                thumbprint_region = region
                break

        if thumbprint_region:
            min_row, min_col, max_row, max_col = thumbprint_region.bbox
            thumbprint = image_np[min_row:max_row, min_col:max_col]

            try:
                enhanced_thumbprint = enhance_fingerprint(thumbprint)

                # Convert the numpy array back to an image
                enhanced_thumbprint_image = PIL.Image.fromarray(enhanced_thumbprint)
            except Exception as e:
                logging.error(f"Error enhancing thumbprint: {e}")
                return {"message": f"Error enhancing thumbprint: {e}"}

            # Resize the enhanced thumbprint image
            new_width = 200  # Choose the desired width
            new_height = 200  # Choose the desired height
            resized_thumbprint_image = enhanced_thumbprint_image.resize((new_width, new_height))

            # Save the resized thumbprint image to a BytesIO object
            output_buffer = BytesIO()
            resized_thumbprint_image.save(output_buffer, format="JPEG")
            thumbprint_data = output_buffer.getvalue()

            # Save the thumbprint image to the database
            db_thumb = ImageEntry1(thumb=thumbprint_data)
            db.add(db_thumb)
            db.commit()
            logging.info("Thumbprint image stored in the database.")

            # Return the processed thumbprint as a response
            return StreamingResponse(BytesIO(thumbprint_data), media_type="image/jpeg")

        else:
            return {"message": "No thumbprint found in the image."}

    except Exception as e:
        logging.error(f"Error processing fingerprint: {e}")
        db.rollback()  # Rollback the transaction in case of error
        return {"message": f"Error processing fingerprint: {e}"}

@app.get("/get_thumbprint/{thumbprint_id}")
async def get_thumbprint(thumbprint_id: int, db: Session = Depends(get_db)):
    try:
        db_thumb = db.query(ImageEntry1).filter(ImageEntry1.id == thumbprint_id).first()
        if db_thumb:
            thumbprint_data = db_thumb.thumb
            return StreamingResponse(BytesIO(thumbprint_data), media_type="image/jpeg")
        else:
            return {"message": "Thumbprint not found."}
    except Exception as e:
        logging.error(f"Error retrieving thumbprint: {e}")
        return {"message": f"Error retrieving thumbprint: {e}"}

# if _name_ == "_main_":
#     uvicorn.run(app, host="127.0.0.1", port=8000)

# Run the FastAPI server
# if _name_ == "_main_":
#     uvicorn.run(app, host="127.0.0.1", port=8000)



# #####-------------------------------------------------- pattern -------------------------------------------------------#####


from models import CapturedImage2
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
# import dlib
import io
# from PIL import Image
import PIL.Image
from db import SessionLocal, engine
import uvicorn
import numpy as np
import cv2
import math
import keras_ocr




# app = FastAPI()

# Initialize the database
CapturedImage2.metadata.create_all(bind=engine)

# Initialize face detector and other variables
# face_detector = dlib.get_frontal_face_detector()
# capture_completed = False  # Flag to indicate if the capture has been completed

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

def highlight_edges(image, color):
    edges = cv2.Canny(image, threshold1=30, threshold2=70)
    highlighted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    highlighted[edges > 0] = color
    return highlighted

def midpoint(x1, y1, x2, y2):
    x_mid = (x1 + x2) // 2.0
    y_mid = (y1 + y2) // 2.0
    return int(x_mid), int(y_mid)

def inpaint_text(img_path, pipeline):
    img = keras_ocr.tools.read(img_path)
    prediction_groups = pipeline.recognize([img])

    
    mask = np.zeros(img.shape[:2], dtype="uint8")
    inpainted_img = img.copy()
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)


    return inpainted_img
def process_and_save_image(input_img_path, db):
    pipeline = keras_ocr.pipeline.Pipeline()

    img_text_removed = inpaint_text(input_img_path, pipeline)

    # Convert NumPy array to PIL Image
    img_text_removed = PIL.Image.fromarray(cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2RGB))

    # Resize the image while maintaining aspect ratio
    max_width = 200  # You can adjust this value based on your requirements
    img_text_removed.thumbnail((max_width, max_width))

    # Convert the resized PIL Image back to NumPy array
    img_text_removed = cv2.cvtColor(np.array(img_text_removed), cv2.COLOR_RGB2BGR)

    # Rest of the code remains the same
    gray = cv2.cvtColor(img_text_removed, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=33, sigmaY=33)
    divide = cv2.divide(gray, blur, scale=255)
    red_color = (0, 0, 255)
    edge_highlighted = highlight_edges(divide, red_color)

    # Convert processed images to binary data
    image_data = cv2.imencode('.jpg', edge_highlighted)[1].tobytes()

    # Update the database with image data
    db_entry = CapturedImage2(
        image_data=image_data,
    )

    db.add(db_entry)
    db.commit()


@app.post("/pattern/")
async def capture_pattern(db: Session = Depends(get_db)):
    global capture_completed

    #Reset capture_completed flag to False
    capture_completed = False

    

    # Simulate image capture from a file
    image_path = 'card_image.jpg'
    img = PIL.Image.open(image_path)
    img_gray = img.convert('L')

    image_bytes = io.BytesIO()


    # Process and save the captured image
    process_and_save_image(image_path, db)

    return {"message": "Pattern Captured, Stored, and Processed"}


@app.get("/pattern_image/{image_id}")
async def get_pattern_image(image_id: int, db: Session = Depends(get_db)):
    try:
        db_entry = db.query(CapturedImage2).filter(CapturedImage2.id == image_id).first()
        if db_entry:
            image_data = db_entry.image_data
            image_np = np.frombuffer(image_data, dtype=np.uint8)
            image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            return StreamingResponse(BytesIO(cv2.imencode('.jpg', image_cv2)[1].tobytes()), media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Pattern image not found")
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
# Run the FastAPI server
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)




# ################################################################ TEXT - DANI  #################################################################





import easyocr
from PIL import Image, ImageFilter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv
import numpy as np
from sqlalchemy import Column, Integer, LargeBinary, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import urllib.parse
from fastapi import FastAPI, Depends, status, HTTPException, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, Depends, status,HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import models
from db import engine, SessionLocal
import json
import db
# Constants for PAN card dimensions (in mm)
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Function to calculate the distance in millimeters
def calculate_distance(coord1, coord2, dimension):
    return abs(coord1 - coord2) / dimension * CARD_WIDTH_MM

# FastAPI app
# app = FastAPI()

# # Encode the password properly
# encoded_password = urllib.parse.quote_plus('Kiran@123')


# DATABASE_URL = "mysql+pymysql://root:encoded_password@localhost:3306/teama_db"  # Replace with your database URL
# engine = create_engine(DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

    
# app = FastAPI()
models.Base.metadata.create_all(bind=engine)
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    id: int
    text: str

# ... (other code)

# class UserBase(BaseModel):

#     text: bytes

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# Load font styles from the CSV file
font_samples = []
with open("fonts data set.txt", "r") as font_file:
    csv_reader = csv.reader(font_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        font_samples.append(row[1])  # Assuming the font style is in the second column (index 1)

# Creating a dataset for training
vectorizer = CountVectorizer().fit(font_samples)
X = vectorizer.transform(font_samples)

# Train a simple Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, font_samples)
@app.post("/analyze_image/", status_code=status.HTTP_200_OK)
async def analyze_image(image_file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Open and preprocess the uploaded image with a filter
    image = PIL.Image.open(image_file.file)
    filtered_image = image.filter(ImageFilter.CONTOUR)  # You can try other filters as well

    # Convert the PIL Image to a NumPy array
    image_array = np.array(filtered_image)

    # Perform OCR on the filtered image using EasyOCR
    results = reader.readtext(image_array)  # Use the image array as input

    # Extracted text
    extracted_text = ' '.join([text for (_, text, _) in results])

    # Get image dimensions
    image_width, image_height = image.size

    # Predict the font style of the extracted text
    extracted_text = extracted_text.replace("\n", " ")  # Remove line breaks
    extracted_text = extracted_text.strip()  # Remove leading/trailing spaces
    text_vector = vectorizer.transform([extracted_text])
    predicted_font = classifier.predict(text_vector)[0]

    # Extracted information
    analysis_output = ""
    analysis_output += f"Extracted Text:\n{extracted_text}\n\n"
    
    for (bbox, text, prob) in results:
        width_ratio = image_width / bbox[2][0]  # To account for EasyOCR's relative coordinates
        text_size = calculate_distance(bbox[0][0], bbox[2][0], image_width)
        word_distance = calculate_distance(bbox[0][0], bbox[2][1], image_height)  # Use bbox[2][1] for word distance
        char_spacing = text_size / len(text)
        font_style = predicted_font

        analysis_output += f"Word: {text}\n"
        analysis_output += f"Text Size (mm): {round(text_size, 2)}\n"
        analysis_output += f"Word Distance (mm): {round(word_distance, 2)}\n"
        analysis_output += f"Character Spacing (mm): {round(char_spacing, 2)}\n"
        analysis_output += f"Font Style: {font_style}\n"
        analysis_output += "=" * 20 + "\n"
    
    # Calculate the dimensions of the card
    card_width_mm = CARD_WIDTH_MM
    card_height_mm = CARD_HEIGHT_MM

    analysis_output += f"Card Dimensions (mm): {card_width_mm:.2f} x {card_height_mm:.2f}\n"

    # Save the analysis output to a file
    output_file_path = 'analysis_output.txt'
    with open(output_file_path, 'w') as output_file:
        output_file.write(analysis_output)

    # Store the analysis_output JSON in the database
    json_response = {"analysis_output": analysis_output}
    json_response_str = json.dumps(json_response)  # Convert JSON dict to a string

    # Create a new YourModel instance and insert it into the database
    db_entry = models.AnalysisResult(text=json_response_str)
    db.add(db_entry)
    db.commit()

    return JSONResponse(content=json_response)

import models  # Import models from the same package
from fastapi.responses import JSONResponse

@app.get("/analysis_results/{analysis_result_id}")
async def get_analysis_result(analysis_result_id: int, db: Session = Depends(get_db)):
    analysis_result = db.query(models.AnalysisResult).filter(models.AnalysisResult.id == analysis_result_id).first()
    if analysis_result is None:
        raise HTTPException(status_code=404, detail="AnalysisResult not found")

    # Convert the SQLAlchemy model instance to a dictionary
    result_dict = {
        "id": analysis_result.id,
        "text": analysis_result.text,
        # Add other attributes as needed
    }

    return JSONResponse(content=result_dict)

#################################################  signature - Tharuni ####################################


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
from sqlalchemy.orm import Session
# from PIL import Image as PILImage
import PIL.Image
from io import BytesIO
import io
# app = FastAPI()

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




# Define a route to receive a POST request to start signature detection
@app.post("/detect_signature/")
async def start_detection(db: Session = Depends(get_db), image_path: str = None):
    image_path='card_image.jpg'
    if image_path is None:
        raise HTTPException(status_code=400, detail='Image path not provided')

    # Load the image from the specified path
    try:
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Image file not found')

    image_pil = PIL.Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image_pil)

    # Resize the captured image (adjust dimensions as needed)
    resized_image = cv2.resize(image_np, (800, 600))

    input_tensor = tf.convert_to_tensor(np.expand_dims(resized_image, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = resized_image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False)

    # Save the image with detections
    image_name = f"detected_image.jpg"
    cv2.imwrite(image_name, image_np_with_detections)
    
    try:
        db_signature = Image(
            image=open('detected_image.jpg', 'rb').read()
        )
        db.add(db_signature)
        db.commit()
    except Exception as e:
        db.rollback()
        # Log the error for debugging purposes
        print(f"Error saving image to the database: {e}")
        raise HTTPException(status_code=500, detail='Error saving image')

    # Return a response indicating success
    return {"message": "Signature detection completed and image saved."}




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


































































































































































































# @app.get('/color_entries_1/{entry_id}', status_code=status.HTTP_200_OK)
# async def get_color_entry(entry_id: int, db: Session = Depends(get_db)):
#     db_entry = db.query(models.ColorEntry1).filter(models.ColorEntry1.id == entry_id).first()
#     if db_entry is None:
#         raise HTTPException(status_code=404, detail='Color entry not found')
    
#     image_data = db_entry.image
#     color_data = db_entry.color_data
    
#     # Convert color data to a dictionary if it's in JSON format
#     color_dict = json.loads(color_data)
    
#     # Create a response JSON containing image URL and color data
#     response_data = {
#         "image_url": f"http://your_server_domain/color_entries_1/image/{entry_id}",
#         "color_data": color_dict
#     }
    
#     # Send the image as a streaming response
#     image_response = StreamingResponse(io.BytesIO(image_data), media_type='image/jpeg')
    
#     # Send the response JSON along with the image
#     return JSONResponse(content=response_data, media_type="application/json")

# @app.get('/color_entries_1/{entry_id}', response_model=ColorEntry1Response, status_code=status.HTTP_200_OK)
# async def get_color_entry(entry_id: int, db: Session = Depends(get_db)):
#     db_entry = db.query(models.ColorEntry1).filter(models.ColorEntry1.id == entry_id).first()
#     if db_entry is None:
#         raise HTTPException(status_code=404, detail='Color entry not found')
    
#     # Convert the data into the Pydantic response model
#     response_entry = ColorEntry1Response(
#         id=db_entry.id,
#         image=db_entry.image,
#         color_data=db_entry.color_data
#     )
    
#     return response_entry

# @app.get('/color_entries_1/{entry_id}', status_code=status.HTTP_200_OK)
# async def get_color_entry(entry_id: int, db: Session = Depends(get_db)):
#     db_entry = db.query(models.ColorEntry1).filter(models.ColorEntry1.id == entry_id).first()
#     if db_entry is None:
#         raise HTTPException(status_code=404, detail='Color entry not found')
    
#     image_data = db_entry.image
#     color_data = db_entry.color_data
    
#     return StreamingResponse(io.BytesIO(image_data), media_type='image/jpeg')
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='127.0.0.1', port=8000)






# @app.post('/process')
# async def process_card_region(image: UploadFile = File(...), db: Session = Depends(get_db)):
#     # Call the capture_frames() function to capture the card region
#     card_region = capture_frames()

#     # Save the card image
#     filename = 'card_image.jpg'  # You can generate a unique filename
#     with open(filename, 'wb') as f:
#         f.write(image.file.read())

#     color_data = exact_color(filename, 900, 12, 2.5)
    
#     # Convert color data to dictionary format {"color_code": "percentage"}
#     color_dict = {entry['c_code']: str(round(entry['percentage'], 2)) for entry in color_data}
    
#     # Save image and color data to the database (using ColorEntry1 class)
#     try:
#         db_color_entry = ColorEntry1(
#             image=image.file.read(),  # Store the image as BLOB data
#             color_data=json.dumps(color_dict)  # Convert to JSON format
#         )
#         db.add(db_color_entry)
#         db.commit()
        
#         return {'color_data': color_dict}
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=500, detail='Error saving image and color data')


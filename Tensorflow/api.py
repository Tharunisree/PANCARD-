import io
import os
import subprocess
import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils
from sqlalchemy.orm import Session
from pydantic import BaseModel
from db import SessionLocal, engine  # Import your database setup
from moduls import SignatureDetectionResult
from Tutorial import signature

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SignatureDetectionResultInput(BaseModel):
    image: bytes

class SignatureDetectionResultOutput(BaseModel):
    id: int
    image_url: str

@app.post("/signature_detection_results/", response_model=SignatureDetectionResultOutput)
async def create_signature_detection_result(
    result_input: SignatureDetectionResultInput,
    db: Session = Depends(get_db)
):
    # Save the uploaded image
    image_bytes = result_input.image
    with open('uploaded_image.jpg', 'wb') as f:
        f.write(image_bytes)

    # Process the image with the signature function
    sign_region = signature()  # You need to implement this function

    # Save the processed image
    cv2.imwrite('detected_signature.jpg', sign_region)

    # Store the result in the database
    result_entry = SignatureDetectionResult(image='detected_signature.jpg')
    db.add(result_entry)
    db.commit()
    db.refresh(result_entry)

    # Return the result to the client
    result_output = SignatureDetectionResultOutput(id=result_entry.id, image_url='detected_signature.jpg')
    return result_output

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
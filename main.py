# main.py
# api/authentication.py
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

# Define an API key scheme
API_KEY = "123456secret"  # Example API key, replace it securely
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def validate_api_key(api_key: str = Security(api_key_header)):
    """Validate the API key sent in the headers."""
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403, 
            detail="Invalid or missing API key."
        )
    return api_key


import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Video Analsis API")

import cv2
import json
from ultralytics import YOLO
from easyocr import Reader
from datetime import timedelta, datetime
from collections import defaultdict
import numpy as np

# Settings
#video_path = s.video_path  # Path to your video
#output_file = s.output_file  # Path to save output JSON
model_path = "yolo11n.pt"  # YOLO model path
vehicle_classes = ["car", "motorcycle", "bus", "truck", "airplane"]  # Vehicle types
person_class = "person"  # Class name for people

# Initialize the model
model = YOLO(model_path)

# Initialize OCR for license plates
reader = Reader(["en"])

def time_to_seconds(time_str):
    """Convert time in format HH:MM:SS to seconds."""
    time_obj = datetime.strptime(time_str, "%H:%M:%S")
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second

def detect_license_plate(crop, reader):
    """Detect license plate using OCR."""
    try:
        results = reader.readtext(crop)
        if results:
            return results[0][-2]  # First result
    except Exception:
        pass
    return None

def detect_dominant_color(crop):
    """Detect dominant color."""
    try:
        data = np.reshape(crop, (-1, 3))
        data = np.float32(data)
        _, labels, palette = cv2.kmeans(data, 1, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
        dominant_color = palette[0].astype(int)
        return tuple(map(int, dominant_color))
    except Exception:
        pass
    return None

def consolidate_objects(data):
    """Consolidate the objects in the JSON data."""
    consolidated = []
    seen = {}

    for obj in data:
        # Skip objects where start_time == end_time
        if obj["start_time"] == obj["end_time"]:
            continue

        # Handle duplicates appearing at the same time
        obj_key = (obj["type"], obj["start_time"], obj["end_time"], obj.get("license_plate"))
        if obj_key in seen:
            existing = seen[obj_key]
            existing["end_time"] = obj["end_time"]
            if obj["license_plate"] != "Not available":
                existing["license_plate"] = obj["license_plate"]
            continue

        # Check for objects with same license plate, same timings, and motion
        duplicate_found = False
        for existing in consolidated:
            if (
                existing["type"] == obj["type"] and
                existing.get("license_plate") == obj.get("license_plate") and
                existing["start_time"] == obj["start_time"] and
                existing["end_time"] == obj["end_time"] and
                existing["motion"] == obj["motion"]
            ):
                duplicate_found = True
                break

        if duplicate_found:
            continue

        # Add unique objects
        consolidated.append(obj)
        seen[obj_key] = obj

    return consolidated

def process_video(video_path, model, reader, output_file):
    """Process the video and save results to a JSON file."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print(f"Error: FPS is zero. Unable to process video: {video_path}")
        return

    duration = frame_count / fps
    print(f"Video Properties: FPS={fps}, Frame Count={frame_count}, Duration={duration} seconds")

    object_data = {}
    track_history = defaultdict(list)

    results = model.track(source=video_path, persist=True, verbose=False, stream=True, imgsz=640)  # YOLO tracking

    frame_number = 0  # Initialize frame counter

    for result in results:
        frame = result.orig_img
        frame_number += 1  # Increment frame counter
        timestamp = str(timedelta(seconds=frame_number / fps)).split(".")[0]  # Time in HH:MM:SS format

        seen_objects = defaultdict(int)  # Track counts of objects in the frame

        for obj in result.boxes:
            cls_id = int(obj.cls.item())
            cls_name = model.names[cls_id]

            if cls_name not in vehicle_classes and cls_name != person_class:
                continue

            # Check if obj.id is None
            if obj.id is None:
                print(f"Warning: Object ID is None for class {cls_name}. Removing from processing...")
                continue

            object_id = f"v{int(obj.id.item())}" if cls_name in vehicle_classes else f"h{int(obj.id.item())}"
            bbox = obj.xyxy[0].tolist()  # Bounding box coordinates
            bbox = list(map(int, bbox))  # Convert to int
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # If the object is not already tracked, initialize it
            if object_id not in object_data:
                object_data[object_id] = {
                    "id": object_id,
                    "type": cls_name,
                    "start_time": timestamp,
                    "end_time": timestamp,  # Initialize end_time as start_time
                    "license_plate": None if cls_name == person_class else "Not available",
                    "color": None,
                    "motion": False,
                    "same_type_count": 0,
                    "bbox": None
                }

            # Update end time
            object_data[object_id]["end_time"] = timestamp

            # Check for motion
            if object_id in track_history and track_history[object_id]:
                last_x, last_y = track_history[object_id][-1]
                if abs(center_x - last_x) > 5 or abs(center_y - last_y) > 5:  # Threshold for motion
                    object_data[object_id]["motion"] = True

            # Add the current coordinates to track history
            track_history[object_id].append((center_x, center_y))

            # Detect license plate or color
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if cls_name in vehicle_classes and object_data[object_id]["license_plate"] == "Not available":
                license_plate = detect_license_plate(crop, reader)
                if license_plate:
                    object_data[object_id]["license_plate"] = license_plate

            if cls_name == person_class and object_data[object_id]["color"] is None:
                color = detect_dominant_color(crop)
                if color:
                    object_data[object_id]["color"] = color

            # Update bbox
            object_data[object_id]["bbox"] = [(x1, y1), (x2, y2)]

            # Update count of same-type objects
            seen_objects[cls_name] += 1

        # Update same-type counts for all seen objects
        for obj_id, obj_info in object_data.items():
            obj_cls = obj_info["type"]
            obj_info["same_type_count"] = seen_objects[obj_cls]

        # Calculate and yield progress
        progress = (frame_number / frame_count) * 100
        yield progress

    # Prepare data for saving
    raw_data = [
        {
            "id": v["id"],
            "type": v["type"],
            "start_time": v["start_time"],
            "end_time": v["end_time"],
            "license_plate": v["license_plate"],
            "color": v["color"],
            "motion": v["motion"],
            "same_type_count": v["same_type_count"],
            "bbox": v["bbox"],
        }
        for v in object_data.values()
    ]

    # Consolidate data to remove duplicates
    result_data = consolidate_objects(raw_data)

    # Save the data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)

    print(f"Processing completed. Results saved to {output_file}")



from fastapi import APIRouter, WebSocket, File, UploadFile, Depends, HTTPException
import shutil
import os
import asyncio
from typing import Dict
from pathlib import Path # Import the video processing function from your algorithm
router = APIRouter()

# Define directories relative to the current file's location
BASE_DIR = Path(__file__).resolve()
UPLOAD_DIR = BASE_DIR / "data" / "raw"  # Directory to save uploaded videos
RESULTS_DIR = BASE_DIR / "data" / "results"  # Directory to save processed JSON results

# Ensure the upload and results directories exist, create if not
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

progress_tracker: Dict[str, float] = {}  # Dictionary to track progress per file


# Endpoint to upload video first
@router.post("/upload/", summary="Upload a video file")
async def upload_video(
        file: UploadFile = File(...),
):
    """
    Endpoint to receive a video file.

    - **file**: The video file being uploaded.
    """
    # Validate file type
    allowed_extensions = ["mp4", "avi", "mov", "mkv"]
    file_extension = file.filename.split(".")[-1]
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Save the file in the upload directory
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Video uploaded successfully", "filename": file.filename}


# Now the WebSocket connection starts after the video is uploaded
@router.websocket("/process/")
async def process_video_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for processing video files.

    - Waits for the video file to be uploaded before establishing the WebSocket connection.
    - Once a video is uploaded, it starts processing the video and sends progress updates.
    - Sends the final JSON file after processing.
    """
    # Wait for the client to upload the video before allowing WebSocket connection
    await websocket.accept()

    try:

        # Now we wait for the client to send the filename through WebSocket after upload
        data = await websocket.receive_json()
        filename = data.get("filename")
        print(f"Received filename: {filename}")
        if not filename:
            await websocket.send_json({"error": "Filename not provided. Please upload a file first."})
            await websocket.close()
            return

        # Check if the file exists in the upload directory
        file_location = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_location):
            await websocket.send_json({"error": f"File {filename} not found."})
            await websocket.close()
            return
        print(f"File found at: {file_location}")
        # Prepare result file path (where to store the JSON output)
        result_file = Path(RESULTS_DIR) / f"{Path(filename).stem}_results.json"

        
            

        # Initialize progress tracking for this file
        progress_tracker[filename] = 0.0
        print("Starting video processing...")
        pro = p.process_video(file_location, p.model, p.reader, str(result_file))
        # Wait for the background task (video processing) to finish
        print("Processing video...")
        for i in pro:
            progress_tracker[filename] = i
            # Send progress update to the client without blocking the loop
            await websocket.send_json({"progress": i})
            await asyncio.sleep(0.001)  # Add a small delay to ensure the client receives the updates
            print(i)

        progress_tracker[filename] = 100.0  # Update progress to 100% after processing is complete
        print("Processing complete.")

        asyncio.sleep(1)  # Wait for a second before sending the final progress update
        # Send final JSON result file path after processing is complete
        await websocket.send_json({"progress": 100.0, "result_file": str(result_file)})

    except Exception as e:
        # Handle any errors that occur during the WebSocket communication
        await websocket.send_json({"error": str(e)})
    finally:
        # Clean up progress tracker after processing is complete or if an error occurs
        if filename and filename in progress_tracker:
            del progress_tracker[filename]
        await websocket.close()










# Include API router
app.include_router(router, prefix="/api", tags=["Video Upload"])

uvicorn.run(app,host='0.0.0.0',port=8000)

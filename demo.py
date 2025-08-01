# File: /content/drive/MyDrive/MiVOLO/demo.py

import logging
import os
import uuid  # For unique filenames
import sys
import shutil # For potential file operations like deleting folders
import threading # For live stream management
import time # For stream delay
import base64 # For encoding/decoding Base64 image data
import numpy as np # For converting base64 to OpenCV image
from collections import deque # For buffering frames

import cv2
import torch
import yt_dlp
# Changed from render_template_string to render_template
from flask import Flask, request, render_template, send_file, url_for, redirect, flash, Response, jsonify, send_from_directory # Import send_from_directory
from torch.nn import Sequential
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules import C2f # Example of another one you might need
from ultralytics.nn.tasks import DetectionModel

from mivolo.data.data_reader import InputType, get_input_type # Ensure InputType is imported

from mivolo.predictor import Predictor # Assuming this path is correct based on previous conversations

# --- Ultralytics Torch Load Patching (as provided by user) ---
import ultralytics.nn.tasks as ultralytics_tasks

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    elif kwargs["weights_only"] is True:
        print("WARNING: weights_only=True was explicitly requested, but overriding to False for compatibility.")
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

ultralytics_tasks.torch_safe_load = lambda weight_path: (_patched_torch_load(weight_path, map_location="cpu"), weight_path)

torch.serialization.add_safe_globals([DetectionModel])
torch.serialization.add_safe_globals([Sequential])
torch.serialization.add_safe_globals([Conv])
torch.serialization.add_safe_globals([C2f])

# --- Logging Setup ---
_logger = logging.getLogger("flask_inference")
if _logger.hasHandlers():
    _logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.INFO)

# --- Flask App Configuration ---
app = Flask(__name__, template_folder='templates') # Tell Flask where to find templates
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_flask'
# IMPORTANT: Change this to a strong, random key in production!
app.secret_key = 'super_secret_key_for_demo_purposes_change_me'

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --- MiVOLO Model Configuration (Global, set at app startup) ---
DETECTOR_WEIGHTS = os.environ.get("DETECTOR_WEIGHTS", "models/yolov8x_person_face.pt")
CHECKPOINT = os.environ.get("CHECKPOINT", "models/model_imdb_cross_person_4.22_99.46.pth.tar")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

WITH_PERSONS_AT_STARTUP = os.environ.get("WITH_PERSONS", "False").lower() == "true"
DISABLE_FACES_AT_STARTUP = os.environ.get("DISABLE_FACES", "False").lower() == "true"
DRAW_OUTPUT_AT_STARTUP = os.environ.get("DRAW_OUTPUT", "True").lower() == "true"

# Define the default tracker config path
DEFAULT_ULTRALYTICS_TRACKER_PATH = os.environ.get("ULTRALYTICS_TRACKER_PATH", "trackers/botsort.yaml") # Default to botsort.yaml

# Global predictor instance (loaded once when the app starts)
predictor = None

def load_predictor():
    """Loads the MiVOLO Predictor model globally when the app starts."""
    global predictor
    if predictor is None:
        _logger.info("Attempting to load MiVOLO Predictor...")

        class PredictorConfigArgs:
            def __init__(self):
                self.detector_weights = DETECTOR_WEIGHTS
                self.checkpoint = CHECKPOINT
                self.device = DEVICE
                self.with_persons = WITH_PERSONS_AT_STARTUP
                self.disable_faces = DISABLE_FACES_AT_STARTUP
                self.draw = DRAW_OUTPUT_AT_STARTUP
                self.output = app.config['OUTPUT_FOLDER']
                # Pass the tracker config here
                self.ultralytics_tracker_config = DEFAULT_ULTRALYTICS_TRACKER_PATH

        args = PredictorConfigArgs()
        try:
            if not os.path.exists(DETECTOR_WEIGHTS):
                _logger.error(f"Detector weights file not found: {DETECTOR_WEIGHTS}")
                raise FileNotFoundError(f"Detector weights not found at {DETECTOR_WEIGHTS}")
            if not os.path.exists(CHECKPOINT):
                _logger.error(f"MiVOLO checkpoint file not found: {CHECKPOINT}")
                raise FileNotFoundError(f"MiVOLO checkpoint not found at {CHECKPOINT}")
            # Check if the tracker config file exists
            if not os.path.exists(DEFAULT_ULTRALYTICS_TRACKER_PATH):
                _logger.warning(f"Ultralyics tracker config file not found: {DEFAULT_ULTRALYTICS_TRACKER_PATH}. Tracking might be disabled or use default Ultralytics settings.")
                # Option to set to None if you want to explicitly disable tracking if config is missing
                # args.ultralytics_tracker_config = None

            predictor = Predictor(args, verbose=True)
            _logger.info("MiVOLO Predictor loaded successfully.")
            _logger.info(f"Predictor configured with: Persons={WITH_PERSONS_AT_STARTUP}, Faces Disabled={DISABLE_FACES_AT_STARTUP}, Device={DEVICE}, Tracker Config={DEFAULT_ULTRALYTICS_TRACKER_PATH}")
        except Exception as e:
            _logger.error(f"Failed to load MiVOLO Predictor. Error: {e}", exc_info=True)
            predictor = None

# Load the model when the app starts within the application context
with app.app_context():
    load_predictor()

# --- Live Stream Variables for Colab Webcam ---
# This queue will hold frames received from the client-side JavaScript
frame_queue = deque(maxlen=5) # Buffer for a few frames
is_streaming_webcam = False
stream_thread = None # To hold the thread that processes frames from the queue
stream_lock = threading.Lock() # For protecting access to is_streaming_webcam and frame_queue

# Global variable for external video capture (RTSP/IP Cam)
video_capture_object_external = None # Moved up for better scope management

# --- Utility Functions ---
def get_direct_video_url(video_url):
    ydl_opts = {
        "format": "bestvideo[ext=mp4]/best",
        "quiet": True,
        "noplaylist": True,
        "no_warnings": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            if "url" in info_dict:
                direct_url = info_dict["url"]
                resolution = (info_dict.get("width"), info_dict.get("height"))
                fps = info_dict.get("fps")
                yid = info_dict.get("id")
                return direct_url, resolution, fps, yid
    except Exception as e:
        _logger.error(f"Error getting YouTube direct URL for {video_url}: {e}")
    return None, None, None, None

def get_local_video_info(vid_uri):
    cap = cv2.VideoCapture(vid_uri)
    if not cap.isOpened():
        _logger.error(f"Failed to open video source {vid_uri}")
        return None, None
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return res, fps

# --- Modified generate_frames for Colab Webcam ---
def generate_processed_frames_from_queue():
    """
    Generates processed frames by taking them from the global frame_queue.
    This function is used for the Colab webcam streaming.
    """
    global is_streaming_webcam

    if predictor is None:
        _logger.error("Predictor not loaded for live stream.")
        with stream_lock: # Ensure thread-safe update
            is_streaming_webcam = False
        # Return an error image
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            cv2.imencode('.jpg', cv2.putText(np.zeros((480, 640, 3), np.uint8), "Model Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2))[1].tobytes() +
            b'\r\n'
        )
        return

    _logger.info("Starting to generate processed frames from queue.")
    while True: # Loop indefinitely, relying on `is_streaming_webcam` for control
        with stream_lock: # Protect access to the flag
            if not is_streaming_webcam:
                _logger.info("Processed frame generation loop stopping (is_streaming_webcam is False).")
                break # Exit the loop if streaming is stopped

        if len(frame_queue) > 0:
            frame_data = frame_queue.popleft()

            # Decode Base64 image data
            try:
                # Remove data:image/jpeg;base64, prefix if present
                if "base64," in frame_data:
                    frame_data = frame_data.split("base64,")[1]

                nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    _logger.warning("Failed to decode frame from Base64. Skipping frame.")
                    continue

                # Process frame with MiVOLO Predictor using `recognize`
                try:
                    _detected_objects, annotated_frame = predictor.recognize(frame)
                    frame_to_encode = annotated_frame
                except Exception as e:
                    _logger.warning(f"Error during live stream processing with MiVOLO: {e}. Displaying original frame.", exc_info=True)
                    frame_to_encode = frame # Fallback to original frame if error

                ret, buffer = cv2.imencode('.jpg', frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Lower quality JPEG encoding
                frame_bytes = buffer.tobytes()

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )
            except Exception as e:
                _logger.error(f"Error processing frame from queue: {e}", exc_info=True)
        else:
            time.sleep(0.01) # Wait a bit if queue is empty

    _logger.info("Processed frame generation loop ended.")

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main page with upload and live stream options."""
    if predictor is None:
        flash("Error: MiVOLO model failed to load. Please check server logs for details.", "error")

    # Pass configuration data to the HTML template
    return render_template(
        'index.html',
        config={
            'WITH_PERSONS_AT_STARTUP': WITH_PERSONS_AT_STARTUP,
            'DISABLE_FACES_AT_STARTUP': DISABLE_FACES_AT_STARTUP,
            'DEVICE': DEVICE,
            'ULTRALYTICS_TRACKER_PATH': DEFAULT_ULTRALYTICS_TRACKER_PATH,
            'predictor_loaded': predictor is not None
        }
    )

# New route to serve output files
@app.route('/output/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# New route to display processed images
@app.route('/display_image')
def display_image():
    image_filename = request.args.get('filename')
    if not image_filename:
        flash("No image specified to display.", "error")
        return redirect(url_for('index'))
    
    # Check if the file actually exists in the output folder for security/robustness
    if not os.path.exists(os.path.join(app.config['OUTPUT_FOLDER'], image_filename)):
        flash("Processed image not found.", "error")
        return redirect(url_for('index'))

    return render_template('image_display.html', image_url=url_for('uploaded_file', filename=image_filename), filename=image_filename)

# New route to display processed videos
@app.route('/play_video')
def play_video():
    video_filename = request.args.get('filename')
    if not video_filename:
        flash("No video specified to play.", "error")
        return redirect(url_for('index'))

    # Check if the file actually exists in the output folder
    if not os.path.exists(os.path.join(app.config['OUTPUT_FOLDER'], video_filename)):
        flash("Processed video not found.", "error")
        return redirect(url_for('index'))

    return render_template('video_playback.html', video_url=url_for('uploaded_file', filename=video_filename), filename=video_filename)


# --- New endpoint to explicitly start webcam stream processing on the server ---
@app.route('/start_webcam_stream', methods=['POST'])
def start_webcam_stream():
    global is_streaming_webcam
    with stream_lock:
        is_streaming_webcam = True
    _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Server-side webcam stream processing signaled to start.")
    return jsonify({"status": "success", "message": "Webcam stream initiated on server."})


# --- New endpoint to receive frames from client-side JavaScript ---
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global frame_queue, is_streaming_webcam
    # Ensure stream is active before accepting frames
    with stream_lock: # Protect access to the flag
        if not is_streaming_webcam:
            _logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Stream not active, cannot receive frames from client. Dropping frame.")
            return jsonify({"status": "error", "message": "Stream not active, cannot receive frames."}), 400

    if not request.is_json:
        _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Upload frame request not JSON.")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    image_data = data.get('image')

    if not image_data:
        _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No image data found in upload frame request.")
        return jsonify({"status": "error", "message": "No image data found"}), 400

    # Add the Base64 image data to the queue
    with stream_lock: # Protect queue access
        if len(frame_queue) < frame_queue.maxlen:
            frame_queue.append(image_data)
            # _logger.debug(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Frame received and added to queue. Queue size: {len(frame_queue)}")
            return jsonify({"status": "success", "message": "Frame received"})
        else:
            _logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Frame queue full ({frame_queue.maxlen}), dropping frame.")
            return jsonify({"status": "warning", "message": "Frame queue full, frame dropped."}), 200


@app.route('/stop_stream_backend_endpoint')
def stop_stream_backend_endpoint():
    """
    Stops the backend processing for live streams (both webcam and external).
    For webcam, this just signals the processing loop to stop.
    For external streams, this releases the OpenCV VideoCapture object.
    """
    global is_streaming_webcam, video_capture_object_external # Declare global

    with stream_lock:
        is_streaming_webcam = False # Signal the webcam processing loop to stop
        if video_capture_object_external is not None and video_capture_object_external.isOpened():
            video_capture_object_external.release()
            video_capture_object_external = None
            _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] External video stream capture released.")

    flash('Live stream stopped.', 'info')
    _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Stream stop signal sent to backend.")
    return "Stream stop signal sent."


@app.route('/video_feed')
def video_feed():
    """
    Provides the MJPEG stream.
    If 'webcam_colab' is requested, it streams processed frames from the queue (fed by client JS).
    Otherwise, it attempts to open an external RTSP/IP camera stream.
    """
    stream_url = request.args.get('url', type=str)

    if stream_url == 'webcam_colab':
        _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting MJPEG stream for Colab webcam (from queue).")
        return Response(generate_processed_frames_from_queue(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else: # External RTSP/IP camera stream
        _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to start MJPEG stream for external URL: {stream_url}")
        global video_capture_object_external

        with stream_lock:
            # Release existing external camera if any, before starting a new one
            if video_capture_object_external is not None and video_capture_object_external.isOpened():
                video_capture_object_external.release()
                video_capture_object_external = None

            video_capture_object_external = cv2.VideoCapture(stream_url)
            if not video_capture_object_external.isOpened():
                _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Could not open external video stream: {stream_url}")
                # Send an error image
                return Response(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    cv2.imencode('.jpg', cv2.putText(np.zeros((480, 640, 3), np.uint8), "Stream Error: Check URL", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2))[1].tobytes() +
                    b'\r\n',
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                )

        _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started external video stream from: {stream_url}")

        def generate_external_frames():
            global video_capture_object_external
            # Loop while the capture object is open and valid
            while video_capture_object_external and video_capture_object_external.isOpened():
                ret, frame = video_capture_object_external.read()
                if not ret:
                    _logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Failed to grab frame from external stream, stream might have ended or been stopped.")
                    break

                try:
                    _detected_objects, annotated_frame = predictor.recognize(frame)
                    frame_to_encode = annotated_frame
                except Exception as e:
                    _logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error during external stream processing with MiVOLO: {e}. Displaying original frame.", exc_info=True)
                    frame_to_encode = frame # Fallback to original frame if error

                ret, buffer = cv2.imencode('.jpg', frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                )

            # Ensure the capture object is released when the loop ends
            if video_capture_object_external and video_capture_object_external.isOpened():
                video_capture_object_external.release()
                video_capture_object_external = None
            _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] External frame generation loop ended and capture released.")

        return Response(generate_external_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Prediction Endpoint (for file uploads and YouTube URLs) ---
@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        flash("Model not loaded. Please restart the application.", "error")
        _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Prediction attempt with unloaded model.")
        return redirect(url_for('index'))

    file = request.files.get('file')
    youtube_url = request.form.get('youtube_url')

    if not file and not youtube_url:
        flash("No file or YouTube URL provided.", "warning")
        _logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No file or YouTube URL provided for prediction.")
        return redirect(url_for('index'))

    input_type = InputType.UNKNOWN
    input_path = None
    output_filename = None

    if file and file.filename:
        filename = file.filename
        file_extension = os.path.splitext(filename)[1].lower()
        unique_filename = str(uuid.uuid4()) + file_extension
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)

        input_type = get_input_type(input_path)
        _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploaded file: {filename}, Type: {input_type}")

    elif youtube_url:
        input_type = InputType.YOUTUBE
        input_path = youtube_url
        _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] YouTube URL provided: {youtube_url}")

    if input_type == InputType.UNKNOWN:
        flash("Unsupported file type or invalid YouTube URL.", "error")
        _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Unsupported input type: {input_path}")
        if input_path and os.path.exists(input_path):
            os.remove(input_path) # Clean up uploaded file
        return redirect(url_for('index'))

    try:
        if input_type == InputType.IMAGE:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError("Could not read image file.")

            _detected_objects, annotated_image = predictor.recognize(image)

            output_filename = f"output_{str(uuid.uuid4())}.jpg"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, annotated_image)

            flash("Image processed successfully!", "success")
            _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Image processed: {output_path}")
            
            # Redirect to the image display page
            return redirect(url_for('display_image', filename=output_filename))

        elif input_type == InputType.VIDEO or input_type == InputType.YOUTUBE:

            video_source = input_path
            if input_type == InputType.YOUTUBE:
                _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Attempting to get direct URL for YouTube video: {input_path}")
                direct_url, resolution, fps, yid = get_direct_video_url(input_path)
                if not direct_url:
                    flash("Could not get direct video URL from YouTube. Please check the URL.", "error")
                    _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Could not get direct YouTube URL for {input_path}")
                    return redirect(url_for('index'))
                video_source = direct_url
                _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Got direct YouTube URL: {video_source}")

            output_video_filename = f"output_video_{str(uuid.uuid4())}.mp4"
            output_video_path = os.path.join(app.config['OUTPUT_FOLDER'], output_video_filename)

            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                flash(f"Error: Could not open video source: {video_source}", "error")
                _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: Could not open video source: {video_source}")
                return redirect(url_for('index'))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting video processing for {video_source}...")
            
            for detected_objects_history, annotated_frame in predictor.recognize_video(video_source):
                if annotated_frame is not None:
                    out.write(annotated_frame)
                else:
                    _logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Received None for annotated_frame from recognize_video. Skipping frame write.")
            
            cap.release()
            out.release()
            _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Video processing complete. Output saved to {output_video_path}")
            
            flash("Video processed successfully!", "success")
            # Redirect to the video playback page
            return redirect(url_for('play_video', filename=output_video_filename))

    except Exception as e:
        _logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Prediction failed: {e}", exc_info=True)
        flash(f"An error occurred during prediction: {e}", "error")
        return redirect(url_for('index'))
    finally:
        if file and file.filename and input_path and os.path.exists(input_path):
            os.remove(input_path)
            _logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Cleaned up uploaded file: {input_path}")

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
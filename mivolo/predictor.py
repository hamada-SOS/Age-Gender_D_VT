# File: /content/drive/MyDrive/MiVOLO/mivolo/predictor.py

from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector  # Make sure this import is correct
from mivolo.structures import AGE_GENDER_TYPE, PersonAndFaceResult
import logging # Import logging to use _logger

_logger = logging.getLogger(__name__) # Get a logger for this module

class Predictor:
    def __init__(self, config, verbose: bool = False):
        # The Detector now accepts verbose, and crucially, tracker_config.
        # Ensure you pass all arguments that Detector's __init__ expects.
        # Assuming config (which comes from PredictorConfigArgs in demo.py)
        # has detector_weights, device, and ultralytics_tracker_config.
        
        # You may also need to pass conf_thresh and iou_thresh if your Detector's __init__
        # doesn't have hardcoded defaults for them. For now, assuming they are fixed
        # within the Detector class, or you would add them to PredictorConfigArgs.
        
        self.detector = Detector(
            weights=config.detector_weights,
            device=config.device,
            half=True, # Assuming 'half' is always true for MiVOLO or comes from config
            verbose=verbose,
            conf_thresh=0.4, # Assuming default, or pass from config if available (e.g., config.conf_thresh)
            iou_thresh=0.7,  # Assuming default, or pass from config if available (e.g., config.iou_thresh)
            tracker_config=config.ultralytics_tracker_config # Pass the tracker config here!
        ) 
        
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    # --- KEEP THIS METHOD AS IS (NO CHANGE) ---
    def recognize(self, image: np.ndarray, **detector_override_kwargs) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        """
        Performs recognition on an image, allowing detector arguments to be overridden.
        Args:
            image: The input image.
            detector_override_kwargs: Optional keyword arguments to pass directly to the detector's predict method.
                                      E.g., tracker=None to explicitly disable tracking for this specific call.
        Returns:
            A tuple containing detected objects and the annotated image.
        """
        # Pass overrides to the detector's predict method
        # The detector's predict method will ensure tracker=None for static images if specified
        detected_objects: PersonAndFaceResult = self.detector.predict(image, **detector_override_kwargs)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im
    # --- END KEEP THIS METHOD AS IS ---

    # --- KEEP THIS METHOD AS IS (NO CHANGE IN LOGIC FOR THIS FIX) ---
    def recognize_video(self, source: str) -> Generator:
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            _logger.error(f"Failed to open video source {source}")
            raise ValueError(f"Failed to open video source {source}")

        detected_objects_history: Dict[int, List[AGE_GENDER_TYPE]] = defaultdict(list)

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Ensure tqdm is imported and used correctly, especially for Flask where
        # a progress bar might not be directly visible in the console.
        # It's fine for debugging, but in production, you might remove tqdm.
        for _ in tqdm.tqdm(range(total_frames), desc="Processing video frames"):
            ret, frame = video_capture.read()
            if not ret:
                break

            # This calls the 'track' method on the detector.
            # The 'Detector.track' method is now responsible for correctly
            # passing the tracker_config to Ultralytics.
            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)

            current_frame_objs = detected_objects.get_results_for_tracking()
            cur_persons: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[0]
            cur_faces: Dict[int, AGE_GENDER_TYPE] = current_frame_objs[1]

            # add tr_persons and tr_faces to history
            for guid, data in cur_persons.items():
                if None not in data:
                    detected_objects_history[guid].append(data)
            for guid, data in cur_faces.items():
                if None not in data:
                    detected_objects_history[guid].append(data)

            detected_objects.set_tracked_age_gender(detected_objects_history)
            if self.draw:
                frame = detected_objects.plot()
            yield detected_objects_history, frame

        video_capture.release() # Release the video capture object
    # --- END KEEP THIS METHOD AS IS ---
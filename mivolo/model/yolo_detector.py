import os
from typing import Dict, Union, Optional

import numpy as np
import PIL
import torch
from mivolo.structures import PersonAndFaceResult
from ultralytics import YOLO
from ultralytics.engine.results import Results
import logging

_logger = logging.getLogger(__name__)

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

# Removed the dynamic discovery block for DEFAULT_ULTRALYTICS_TRACKER_PATH
# This value will now effectively be None unless explicitly set elsewhere,
# but it's okay because Predictor will always pass the correct path.

class Detector:
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
        # tracker_config parameter now expects to be explicitly provided.
        # Its default is None, indicating no tracker config unless passed.
        tracker_config: Optional[str] = None, # Changed default to None
    ):
        self.yolo = YOLO(weights)
        self.yolo.fuse()

        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.yolo.model = self.yolo.model.half()

        self.detector_names: Dict[int, str] = self.yolo.model.names

        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose}

        self.tracker_config = tracker_config # Store the tracker config path

        if self.tracker_config is None:
            _logger.warning("Detector initialized without a valid tracker config. Video tracking might fail if Ultralytics requires one.")
        elif not os.path.exists(self.tracker_config):
             _logger.error(f"Provided tracker config does not exist: {self.tracker_config}. Video tracking will likely fail.")
             self.tracker_config = None # Set to None if an invalid path was passed

    def predict(self, image: Union[np.ndarray, str, "PIL.Image"], **yolo_predict_kwargs) -> PersonAndFaceResult:
        combined_kwargs = self.detector_kwargs.copy()
        combined_kwargs.update(yolo_predict_kwargs)
        
        # Ensure 'tracker' is not passed to predict unless specifically overridden
        if 'tracker' not in combined_kwargs:
            combined_kwargs['tracker'] = None # Explicitly prevent tracking for .predict() by default

        results: Results = self.yolo.predict(image, **combined_kwargs)[0]
        return PersonAndFaceResult(results)

    def track(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
        if self.tracker_config and os.path.exists(self.tracker_config):
            results: Results = self.yolo.track(image, persist=True, tracker=self.tracker_config, **self.detector_kwargs)[0]
        else:
            _logger.warning("No valid tracker config path provided to Detector. Attempting to run YOLO 'track' without a specific tracker YAML. This may lead to errors or default Ultralytics behavior if a tracker config is mandatory.")
            # If a valid tracker_config is not available, try calling .track() without the 'tracker' argument
            # This will make Ultralytics fall back to its internal default, or fail if it strictly requires one.
            results: Results = self.yolo.track(image, persist=True, **self.detector_kwargs)[0]
            
        return PersonAndFaceResult(results)
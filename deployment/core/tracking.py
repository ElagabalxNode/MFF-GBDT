"""
Tracking module for MVP inference pipeline
Adapts YOLOv8 BoT-SORT/ByteTrack output and applies heuristics filtering
"""

import sys
import os
import cv2
import numpy as np
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class BroilerTracker:
    """
    Wrapper for YOLOv8 built-in tracking results.
    
    Features:
    - Accepts pre-assigned track_ids from YOLO
    - Calculates Solidity (ContourArea / HullArea) for heuristics filtering
    - Filters out frames with spread wings (low solidity)
    """
    
    def __init__(self, min_solidity: float = 0.90, frame_rate: int = 30):
        """
        Initialize tracker wrapper.
        
        Args:
            min_solidity: Minimum solidity threshold (0.0-1.0).
                          If solidity < min_solidity, skip feature extraction
            frame_rate: Not used for YOLO built-in tracker, kept for API compatibility
        """
        self.min_solidity = min_solidity
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.logger.info(
            f"BroilerTracker (YOLO Built-in) initialized: min_solidity={min_solidity}"
        )
    
    def calculate_solidity(self, mask: np.ndarray) -> float:
        """
        Calculate solidity metric: ContourArea / HullArea.
        
        Solidity measures how "solid" the shape is. Low solidity indicates
        spread wings or irregular shape (e.g., < 0.9).
        
        Args:
            mask: Binary mask (uint8, 0 or 255)
            
        Returns:
            Solidity value (0.0-1.0). Returns 0.0 if no contour found.
        """
        if mask is None or mask.size == 0:
            return 0.0
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Get contour with maximum area
        max_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(max_contour)
        
        if contour_area == 0:
            return 0.0
        
        # Calculate convex hull
        hull = cv2.convexHull(max_contour)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
        
        # Solidity = ContourArea / HullArea
        solidity = contour_area / hull_area
        
        return float(solidity)
    
    def update(self, detections: list) -> list:
        """
        Process detections that already have track_ids from YOLO.
        Calculates solidity and sets skip flags.
        
        Args:
            detections: List of detection dicts, each containing:
                - 'box': [x1, y1, x2, y2]
                - 'score': confidence score (float)
                - 'mask': binary mask (np.ndarray)
                - 'track_id': int (optional, -1 if not tracked)
                
        Returns:
            List of tracked objects, same as input but with:
                - 'solidity': calculated solidity (float)
                - 'skip_features': bool (True if solidity < min_solidity)
        """
        if not detections:
            return []
        
        tracked_detections = []
        
        for det in detections:
            # Create a copy to avoid modifying original in place
            tracked_det = det.copy()
            
            # Ensure track_id exists (if not provided by YOLO for some reason)
            if 'track_id' not in tracked_det or tracked_det['track_id'] is None:
                 # If no track ID, we can't use it for weight aggregation properly,
                 # but we still pass it through.
                 # Using -1 to indicate no track.
                 tracked_det['track_id'] = -1
            
            # Calculate solidity if mask available
            if 'mask' in tracked_det:
                solidity = self.calculate_solidity(tracked_det['mask'])
                tracked_det['solidity'] = solidity
                tracked_det['skip_features'] = (
                    solidity < self.min_solidity
                )
            else:
                tracked_det['solidity'] = 1.0  # Assume solid if no mask
                tracked_det['skip_features'] = False
            
            # Only include if it has a valid track ID (optional check, 
            # usually we want everything but maybe filter non-tracked?)
            # For now, we keep everything, but main.py might handle track_id=-1 gracefully 
            # (it creates a new track for every -1 or ignores it?)
            # main.py uses track_id as key. If -1, it would collide.
            # However, YOLO usually assigns IDs. If it doesn't, it means it's a lost track or new detection.
            
            tracked_detections.append(tracked_det)
        
        return tracked_detections


class TrackBuffer:
    """
    Buffer for storing weight predictions per track_id.
    
    Tracks weight predictions for each active track and provides
    aggregation methods (median) when track is lost.
    """
    
    def __init__(self):
        """
        Initialize track buffer.
        
        Stores: track_id -> list of weights
        """
        self.tracks = {}  # Dict[int, List[float]]
        self.track_start_frame = {}  # Dict[int, int] - frame when track started
        self.current_frame = 0
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def set_current_frame(self, frame_id: int):
        """Update current frame ID."""
        self.current_frame = frame_id
    
    def add_weight(self, track_id: int, weight: float):
        """
        Add weight prediction to track buffer.
        
        Args:
            track_id: Track ID
            weight: Predicted weight (kg)
        """
        if track_id == -1:
             # Ignore detections without valid track ID
             return

        if track_id not in self.tracks:
            self.tracks[track_id] = []
            self.track_start_frame[track_id] = self.current_frame
        
        self.tracks[track_id].append(weight)
    
    def get_median_weight(self, track_id: int) -> float:
        """
        Calculate median weight for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Median weight (kg), or 0.0 if track not found or empty
        """
        if track_id not in self.tracks or len(self.tracks[track_id]) == 0:
            return 0.0
        
        weights = self.tracks[track_id]
        return float(np.median(weights))
    
    def get_track_duration(self, track_id: int) -> int:
        """
        Get track duration in frames.
        
        Args:
            track_id: Track ID
            
        Returns:
            Duration in frames (number of weight predictions)
        """
        if track_id not in self.tracks:
            return 0
        
        return len(self.tracks[track_id])
    
    def remove_track(self, track_id: int):
        """
        Remove track from buffer.
        
        Args:
            track_id: Track ID to remove
        """
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.track_start_frame:
            del self.track_start_frame[track_id]
    
    def get_active_tracks(self) -> list:
        """
        Get list of active track IDs.
        
        Returns:
            List of track IDs
        """
        return list(self.tracks.keys())
    
    def has_track(self, track_id: int) -> bool:
        """
        Check if track exists in buffer.
        
        Args:
            track_id: Track ID
            
        Returns:
            True if track exists
        """
        return track_id in self.tracks

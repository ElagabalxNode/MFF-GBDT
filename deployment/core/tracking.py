"""
Tracking module for MVP inference pipeline
Implements ByteTrack-based multi-object tracking with heuristics filtering
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

try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    try:
        from byte_tracker import BYTETracker
    except ImportError:
        try:
            from bytetrack import BYTETracker
        except ImportError:
            raise ImportError(
                "ByteTrack not found. Install with: "
                "pip install byte-track or clone ByteTrack repo"
            )


class BroilerTracker:
    """
    Multi-object tracker for broiler chickens using ByteTrack algorithm.
    
    Features:
    - Assigns persistent track_id to detected objects
    - Calculates Solidity (ContourArea / HullArea) for heuristics filtering
    - Filters out frames with spread wings (low solidity)
    """
    
    def __init__(self, min_solidity: float = 0.90, frame_rate: int = 30):
        """
        Initialize tracker.
        
        Args:
            min_solidity: Minimum solidity threshold (0.0-1.0).
                          If solidity < min_solidity, skip feature extraction
            frame_rate: Camera frame rate (for ByteTrack internal timing)
        """
        self.min_solidity = min_solidity
        self.frame_rate = frame_rate
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ByteTrack
        # ByteTrack parameters (can be tuned):
        # - track_thresh: detection confidence threshold
        # - track_buffer: number of frames to keep lost tracks
        # - match_thresh: IoU threshold for matching
        # - frame_rate: frames per second
        # BYTETracker can be initialized with args object or parameters
        try:
            # Try with parameters (for byte-track package)
            self.tracker = BYTETracker(
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                frame_rate=frame_rate
            )
        except TypeError:
            # Try with args object (for yolox.tracker version)
            class Args:
                def __init__(self, fr):
                    self.track_thresh = 0.5
                    self.track_buffer = 30
                    self.match_thresh = 0.8
                    self.frame_rate = fr
            args = Args(frame_rate)
            self.tracker = BYTETracker(args)
        
        self.logger.info(
            f"BroilerTracker initialized: min_solidity={min_solidity}, "
            f"frame_rate={frame_rate}"
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
        Update tracker with new detections and assign track_ids.
        
        Args:
            detections: List of detection dicts, each containing:
                - 'box': [x1, y1, x2, y2]
                - 'score': confidence score (float)
                - 'mask': binary mask (np.ndarray, optional)
                
        Returns:
            List of tracked objects, each containing:
                - All original detection fields
                - 'track_id': assigned track ID (int)
                - 'solidity': calculated solidity (float)
                - 'skip_features': bool (True if solidity < min_solidity)
        """
        if not detections:
            return []
        
        # Prepare detections for ByteTrack
        # ByteTrack expects: [x1, y1, x2, y2, score] format
        dets = []
        for det in detections:
            box = det['box']
            score = det['score']
            # ByteTrack format: [x1, y1, x2, y2, score]
            dets.append([box[0], box[1], box[2], box[3], score])
        
        dets = np.array(dets, dtype=np.float32)
        
        # Update tracker
        # ByteTrack.update expects (dets, img_info, img_size)
        # img_info and img_size can be None for basic tracking
        try:
            online_targets = self.tracker.update(dets, None, None)
        except TypeError:
            # Some versions only need dets
            online_targets = self.tracker.update(dets)
        
        # Map track_ids back to detections
        tracked_detections = []
        
        for target in online_targets:
            # Extract track_id and bbox from ByteTrack output
            track_id = int(target.track_id)
            # ByteTrack returns tlwh format [x, y, w, h]
            if hasattr(target, 'tlwh'):
                tlwh = target.tlwh
            elif hasattr(target, 'tlbr'):
                # Some versions return tlbr [x1, y1, x2, y2]
                x1, y1, x2, y2 = target.tlbr
                w = x2 - x1
                h = y2 - y1
            else:
                # Fallback: try to get from other attributes
                x1, y1, w, h = target.bbox if hasattr(target, 'bbox') else (0, 0, 0, 0)
            
            if 'x2' not in locals():
                x1, y1, w, h = tlwh
                x2 = x1 + w
                y2 = y1 + h
            
            # Find matching detection by bbox IoU
            best_match_idx = None
            best_iou = 0.0
            
            for idx, det in enumerate(detections):
                det_box = det['box']
                iou = self._calculate_iou(
                    [x1, y1, x2, y2],
                    det_box
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = idx
            
            if best_match_idx is not None:
                # Copy detection data
                tracked_det = detections[best_match_idx].copy()
                tracked_det['track_id'] = track_id
                
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
                
                tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def _calculate_iou(self, box1: list, box2: list) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value (0.0-1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union


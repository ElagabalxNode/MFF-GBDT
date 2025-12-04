"""
Main entrypoint for MVP inference pipeline
Orchestrates: Segmentation → Features → GBDT
"""

import sys
import os
import argparse
import time
import threading
import queue
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deployment.core.segmentation import SegmentationInference
from deployment.core.feature_extract import FeatureExtractor
from deployment.core.predict_weight import WeightPredictor
from deployment.database import InferenceDB

# Tracking imports are done conditionally in live mode only
# to avoid requiring ByteTrack for batch processing


class InferenceConsumer(threading.Thread):
    """
    Stage 2: Inference Core (Consumer)
    
    Works in infinite loop, processing frames from InputQueue.
    Performs: Segmentation → Tracking → Filtering
    """
    
    def __init__(
        self,
        input_queue: queue.Queue,
        config: dict,
        name: str = "InferenceConsumer"
    ):
        """
        Initialize inference consumer thread.
        
        Args:
            input_queue: Queue with frames from camera.
                       Format: (frame_id, rgb_image, depth_image_z16)
            config: Configuration dict
            name: Thread name
        """
        super().__init__(daemon=True, name=name)
        self.input_queue = input_queue
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.running = False
        
        # Initialize segmentation
        seg_config = config.get('segmentation', {})
        self.segmentation = SegmentationInference(
            device=seg_config.get('device'),
            confidence_threshold=seg_config.get('conf_threshold', 0.85),
            config=config
        )
        
        # Initialize tracker (lazy import to avoid requiring ByteTrack in batch mode)
        from deployment.core.tracking import BroilerTracker, TrackBuffer
        
        tracker_config = config.get('tracker', {})
        camera_config = config.get('camera', {})
        frame_rate = camera_config.get('fps', 30)
        min_solidity = tracker_config.get('filters', {}).get('min_solidity', 0.90)
        
        self.tracker = BroilerTracker(
            min_solidity=min_solidity,
            frame_rate=frame_rate
        )
        
        # Initialize feature extractor
        self.logger.info("Loading feature extractor...")
        self.feature_extractor = FeatureExtractor(
            config=config, device=seg_config.get('device')
        )
        
        # Initialize weight predictor
        self.logger.info("Loading weight predictor...")
        self.weight_predictor = WeightPredictor(config=config)
        
        # Initialize track buffer
        self.track_buffer = TrackBuffer()
        
        # Initialize database
        db_config = config.get('database', {})
        db_path = db_config.get('path', 'data/inference_prod.db')
        self.db = InferenceDB(db_path=db_path)
        
        # Camera ID (from config or default)
        self.cam_id = camera_config.get('device_index', 0)
        
        # Track previous active tracks for loss detection
        self.previous_active_tracks = set()
        
        self.logger.info("InferenceConsumer initialized")
    
    def start(self):
        """Start consumer thread."""
        self.logger.info("Starting inference consumer...")
        self.running = True
        super().start()
        self.logger.info("Inference consumer started")
    
    def stop(self):
        """Stop consumer thread."""
        self.logger.info("Stopping inference consumer...")
        self.running = False
        self.join(timeout=2.0)
        self.logger.info("Inference consumer stopped")
    
    def run(self):
        """Main consumer loop - processes frames from input queue."""
        self.logger.info("Inference consumer loop started")
        
        while self.running:
            try:
                # Get frame from input queue (with timeout for graceful shutdown)
                try:
                    frame_data = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                frame_id, rgb_image, depth_image_z16 = frame_data
                
                # Update track buffer current frame
                self.track_buffer.set_current_frame(frame_id)
                
                # Stage 2.1: Segmentation (YOLO)
                instances = self.segmentation.segment_frame(rgb_image)
                
                if not instances:
                    # No detections, check for lost tracks
                    self._process_lost_tracks(set())
                    continue
                
                # Stage 2.2: Tracking & Filtering
                # tracker.update() already calculates solidity and sets skip_features flag
                tracked_instances = self.tracker.update(instances)
                
                # Get current active track IDs
                current_active_tracks = {inst['track_id'] for inst in tracked_instances}
                
                # Stage 3-4: Feature Extraction & Weight Prediction
                for inst in tracked_instances:
                    track_id = inst['track_id']
                    
                    # Skip if features should be skipped (low solidity)
                    if inst.get('skip_features', False):
                        self.logger.debug(
                            f"Frame {frame_id}, track {track_id}: "
                            f"skipping features (solidity={inst.get('solidity', 0):.3f})"
                        )
                        
                        # Debug: Save skipped detection to DB
                        self.db.save_raw_detection(
                            frame_id=frame_id,
                            track_id=track_id,
                            weight=None,
                            score=inst.get('score', 0.0),
                            box=inst.get('box', [0, 0, 0, 0]),
                            solidity=inst.get('solidity', 0.0),
                            features=None
                        )
                        continue
                    
                    try:
                        # Stage 3: Extract features
                        features = self.feature_extractor.extract_all_features(
                            mask=inst['mask'],
                            maskImg=inst['maskImg'],
                            depth_image_z16=depth_image_z16
                        )
                        
                        # Stage 4: Predict weight
                        weight = self.weight_predictor.predict(features)
                        
                        # Stage 5: Add to track buffer
                        self.track_buffer.add_weight(track_id, weight)
                        
                        # Debug: Save raw detection to DB
                        self.db.save_raw_detection(
                            frame_id=frame_id,
                            track_id=track_id,
                            weight=weight,
                            score=inst.get('score', 0.0),
                            box=inst.get('box', [0, 0, 0, 0]),
                            solidity=inst.get('solidity', 0.0),
                            features=features.tolist()
                        )
                        
                        self.logger.debug(
                            f"Frame {frame_id}, track {track_id}: "
                            f"weight={weight:.3f} kg"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error processing track {track_id} in frame {frame_id}: {e}",
                            exc_info=True
                        )
                        # Continue with other tracks
                        continue
                
                # Stage 5: Process lost tracks (tracks that were active before but not now)
                self._process_lost_tracks(current_active_tracks)
                
                # Update previous active tracks
                self.previous_active_tracks = current_active_tracks
                
                self.logger.debug(
                    f"Processed frame {frame_id}: "
                    f"{len(tracked_instances)} tracked instances, "
                    f"{len(current_active_tracks)} active tracks"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error in inference consumer loop: {e}",
                    exc_info=True
                )
                # Continue processing even if one frame fails
                continue
        
        self.logger.info("Inference consumer loop stopped")
    
    def _process_lost_tracks(self, current_active_tracks: set):
        """
        Process tracks that were lost (active before but not in current frame).
        
        Args:
            current_active_tracks: Set of currently active track IDs
        """
        # Find lost tracks (were active before but not now)
        lost_tracks = self.previous_active_tracks - current_active_tracks
        
        for track_id in lost_tracks:
            if not self.track_buffer.has_track(track_id):
                continue
            
            try:
                # Calculate median weight
                median_weight = self.track_buffer.get_median_weight(track_id)
                duration = self.track_buffer.get_track_duration(track_id)
                num_predictions = duration  # Same as duration (one prediction per frame)
                
                # Save to database
                timestamp = datetime.now().isoformat()
                self.db.save_bird_record(
                    timestamp=timestamp,
                    cam_id=str(self.cam_id),
                    track_id=track_id,
                    median_weight=median_weight,
                    duration_in_frames=duration,
                    num_predictions=num_predictions
                )
                
                self.logger.info(
                    f"Track {track_id} lost: "
                    f"median_weight={median_weight:.3f} kg, "
                    f"duration={duration} frames, "
                    f"predictions={num_predictions}"
                )
                
                # Remove from buffer
                self.track_buffer.remove_track(track_id)
                
            except Exception as e:
                self.logger.error(
                    f"Error processing lost track {track_id}: {e}",
                    exc_info=True
                )
                # Still remove from buffer to prevent memory leak
                self.track_buffer.remove_track(track_id)


class MVPInferencePipeline:
    """
    Batch inference pipeline for processing images from files.
    
    NOTE: This class uses the new MLflow-based API but processes static images.
    For real-time inference from camera, use InferenceConsumer instead.
    
    IMPORTANT: Tracking is NOT used in batch mode - each image is processed independently.
    Tracking is only used in live mode (InferenceConsumer) for video streams.
    """
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration dict with MLflow settings (YAML format)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components using new MLflow-based API
        print("Loading models from MLflow...")
        print("NOTE: Tracking is DISABLED in batch mode (not needed for static images)")
        seg_config = config.get('segmentation', {})
        self.segmentation = SegmentationInference(
            device=seg_config.get('device'),
            confidence_threshold=seg_config.get('conf_threshold', 0.85),
            config=config
        )
        
        self.feature_extractor = FeatureExtractor(
            config=config,
            device=seg_config.get('device')
        )
        
        self.weight_predictor = WeightPredictor(config=config)
        
        # Initialize database
        db_config = config.get('database', {})
        db_path = db_config.get('path', 'data/inference_results.db')
        self.db = InferenceDB(db_path=db_path)
        print(f"Database enabled: {db_path}")
        
        print("Pipeline initialized (batch mode - no tracking)")
    
    def process_image(self, image_path: str, depth_image_path: str = None, 
                     save_results: bool = True) -> dict:
        """
        Process single image through full pipeline
        
        Args:
            image_path: Path to RGB image
            depth_image_path: Path to depth map (optional)
            save_results: Whether to save results to database
            
        Returns:
            Dict with results:
                - 'image_path': input image path
                - 'instances': list of instance predictions
                - 'processing_time': total time (seconds)
        """
        import cv2
        import numpy as np
        
        start_time = time.time()
        
        # Load RGB image
        rgb_image = cv2.imread(image_path)
        if rgb_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Load depth image if provided
        depth_image_z16 = None
        if depth_image_path and os.path.exists(depth_image_path):
            depth_image_z16 = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
            if depth_image_z16 is None:
                print(f"Warning: Could not load depth image: {depth_image_path}")
        
        # Stage 1: Segmentation (no tracking in batch mode)
        print(f"Segmenting image: {image_path}")
        
        # Create visualization path
        vis_dir = Path("deployment/data/visualizations")
        vis_dir.mkdir(parents=True, exist_ok=True)
        img_name = Path(image_path).stem
        vis_path = vis_dir / f"{img_name}_segmentation.png"
        
        instances_seg = self.segmentation.segment_frame(
            rgb_image, 
            visualize=True, 
            save_path=str(vis_path)
        )
        print(f"Found {len(instances_seg)} instances (visualization: {vis_path})")
        
        if not instances_seg:
            return {
                'image_path': image_path,
                'instances': [],
                'processing_time': time.time() - start_time,
                'num_instances': 0
            }
        
        # Stage 2 & 3: Features + Weight prediction for each instance
        # NOTE: No tracking here - each instance is processed independently
        instances_results = []
        for i, inst in enumerate(instances_seg):
            try:
                # Extract features
                features = self.feature_extractor.extract_all_features(
                    mask=inst['mask'],
                    maskImg=inst['maskImg'],
                    depth_image_z16=depth_image_z16 if depth_image_z16 is not None else np.zeros_like(rgb_image[:, :, 0], dtype=np.uint16)
                )
                
                # Predict weight
                predicted_weight = self.weight_predictor.predict(features)
                
                instances_results.append({
                    'instance_id': i,
                    'predicted_weight': predicted_weight,
                    'confidence_score': inst.get('score', 0.0),
                    'box': inst.get('box', [0, 0, 0, 0]),
                    'features': features.tolist()  # For debugging/storage
                })
            except Exception as e:
                print(f"Error processing instance {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        results = {
            'image_path': image_path,
            'instances': instances_results,
            'processing_time': processing_time,
            'num_instances': len(instances_results)
        }
        
        # Save to database if enabled
        if save_results:
            try:
                session_id = self.db.save_inference_session(
                    image_path=image_path,
                    instances=instances_results,
                    processing_time=processing_time,
                    config=self.config
                )
                results['session_id'] = session_id
            except Exception as e:
                print(f"Error saving to database: {e}")
        
        return results
    
    def process_batch(self, image_paths: list, depth_image_paths: list = None) -> list:
        """
        Process multiple images
        
        Args:
            image_paths: List of image paths
            depth_image_paths: Optional list of depth image paths
            
        Returns:
            List of results dicts
        """
        results = []
        for i, img_path in enumerate(image_paths):
            depth_path = depth_image_paths[i] if depth_image_paths else None
            result = self.process_image(img_path, depth_path)
            results.append(result)
        return results


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file. If None, uses default path.
        
    Returns:
        Configuration dict
    """
    import yaml
    
    # Get deployment directory
    deployment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default config path
    if config_path is None:
        config_path = os.path.join(deployment_dir, 'config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve relative database path
        if 'database' in config and 'path' in config['database']:
            db_path = config['database']['path']
            if not os.path.isabs(db_path):
                config['database']['path'] = os.path.join(deployment_dir, db_path)
        
        return config
    else:
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Please create config.yaml in deployment/ directory."
        )



def setup_logging(log_dir: str = 'logs'):
    """
    Setup logging with rotation.
    
    Args:
        log_dir: Directory to store logs
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pipeline.log')
    
    # Create handlers
    stream_handler = logging.StreamHandler()
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    
    # Set format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    
    # Ensure file handler flushes immediately
    file_handler.flush()


def main():
    """Main entrypoint"""
    # Setup logging first
    setup_logging()
    
    parser = argparse.ArgumentParser(description='MFF-GBDT MVP Inference Pipeline')
    parser.add_argument('--input', type=str, required=False,
                       help='Input image path or directory (required for batch mode)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--no-db', action='store_true',
                       help='Disable database storage')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (auto-detect if not specified)')
    parser.add_argument('--live', action='store_true',
                       help='Run in live mode with camera')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        if 'segmentation' not in config:
            config['segmentation'] = {}
        config['segmentation']['device'] = args.device

    if args.live:
        # Live Inference Mode
        from deployment.hardware.camera import CameraProducer
        
        # Setup queue
        frame_queue = queue.Queue(maxsize=2)
        
        # Initialize threads
        camera_config = config.get('camera', {})
        producer = CameraProducer(
            input_queue=frame_queue,
            camera_config=camera_config,
            max_queue_size=2
        )
        
        consumer = InferenceConsumer(
            input_queue=frame_queue,
            config=config
        )
        
        try:
            print("Starting Live Inference Pipeline...")
            print("Press Ctrl+C to stop")
            
            producer.start()
            consumer.start()
            
            # Keep main thread alive
            while True:
                time.sleep(1.0)
                if not producer.is_alive() or not consumer.is_alive():
                    print("One of the threads died. Exiting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping pipeline...")
        finally:
            producer.stop()
            consumer.stop()
            print("Pipeline stopped")
            
    else:
        # Batch/File Mode (no tracking - each image processed independently)
        if not args.input:
            parser.error(
                "the following arguments are required: "
                "--input (unless --live is used)"
            )

        print("=" * 60)
        print("BATCH MODE: Processing static images")
        print("Tracking: DISABLED (not needed for static images)")
        print("=" * 60)

        # Initialize pipeline
        pipeline = MVPInferencePipeline(config)

        # Process input
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image
            results = pipeline.process_image(str(input_path))
            print("\nResults:")
            print(f"  Image: {results['image_path']}")
            print(f"  Instances: {results['num_instances']}")
            print(f"  Processing time: {results['processing_time']:.2f}s")
            for i, inst in enumerate(results['instances']):
                weight = inst['predicted_weight']
                conf = inst['confidence_score']
                print(
                    f"  Instance {i+1}: {weight:.3f} kg "
                    f"(confidence: {conf:.2f})"
                )
        elif input_path.is_dir():
            # Batch processing
            image_files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
            results = pipeline.process_batch([str(f) for f in image_files])
            print(f"\nProcessed {len(results)} images")
        else:
            print(f"Error: Input path does not exist: {args.input}")
            return
        
        # Save results if output directory specified
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            results_file = output_path / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()


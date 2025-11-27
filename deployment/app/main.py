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
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deployment.core.segmentation import SegmentationInference
from deployment.core.feature_extract import FeatureExtractor
from deployment.core.predict_weight import WeightPredictor
from deployment.core.tracking import BroilerTracker
from deployment.database import InferenceDB


class InferenceConsumer(threading.Thread):
    """
    Stage 2: Inference Core (Consumer)
    
    Works in infinite loop, processing frames from InputQueue.
    Performs: Segmentation → Tracking → Filtering
    """
    
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        config: dict,
        name: str = "InferenceConsumer"
    ):
        """
        Initialize inference consumer thread.
        
        Args:
            input_queue: Queue with frames from camera.
                       Format: (frame_id, rgb_image, depth_image_z16)
            output_queue: Queue for processed results (for Stage 3).
                        Format: (frame_id, tracked_instances, depth_image)
            config: Configuration dict
            name: Thread name
        """
        super().__init__(daemon=True, name=name)
        self.input_queue = input_queue
        self.output_queue = output_queue
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
        
        # Initialize tracker
        tracker_config = config.get('tracker', {})
        camera_config = config.get('camera', {})
        frame_rate = camera_config.get('fps', 30)
        min_solidity = tracker_config.get('filters', {}).get('min_solidity', 0.90)
        
        self.tracker = BroilerTracker(
            min_solidity=min_solidity,
            frame_rate=frame_rate
        )
        
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
                
                # Stage 2.1: Segmentation (YOLO)
                instances = self.segmentation.segment_frame(rgb_image)
                
                if not instances:
                    # No detections, skip tracking
                    continue
                
                # Stage 2.2: Tracking & Filtering
                tracked_instances = self.tracker.update(instances)
                
                # Stage 2.3: Solidity calculation and filtering
                # (Already done in tracker.update, but ensure all have it)
                for inst in tracked_instances:
                    if 'solidity' not in inst:
                        # Calculate if missing
                        if 'mask' in inst:
                            inst['solidity'] = self.tracker.calculate_solidity(
                                inst['mask']
                            )
                            inst['skip_features'] = (
                                inst['solidity'] < self.tracker.min_solidity
                            )
                        else:
                            inst['solidity'] = 1.0
                            inst['skip_features'] = False
                
                # Put results in output queue for Stage 3 (Feature Extraction)
                try:
                    self.output_queue.put(
                        (frame_id, tracked_instances, depth_image_z16),
                        block=False
                    )
                except queue.Full:
                    self.logger.warning(
                        f"Output queue full, dropping frame {frame_id}"
                    )
                
                self.logger.debug(
                    f"Processed frame {frame_id}: "
                    f"{len(tracked_instances)} tracked instances"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Error in inference consumer loop: {e}",
                    exc_info=True
                )
                # Continue processing even if one frame fails
                continue
        
        self.logger.info("Inference consumer loop stopped")


class MVPInferencePipeline:
    """End-to-end inference pipeline"""
    
    def __init__(self, config: dict):
        """
        Initialize pipeline with configuration
        
        Args:
            config: Configuration dict with paths to models and settings
        """
        self.config = config
        
        # Initialize components
        print("Loading models...")
        self.segmentation = SegmentationInference(
            model_path=config['segmentation']['model_path'],
            device=config.get('device'),
            confidence_threshold=config['segmentation'].get('confidence_threshold', 0.90)
        )
        
        self.feature_extractor = FeatureExtractor(
            resnet_weights_path=config['features'].get('resnet_weights_path'),
            device=config.get('device')
        )
        
        self.weight_predictor = WeightPredictor(
            model_path=config['gbdt']['model_path'],
            model_type=config['gbdt'].get('model_type', 'lightgbm')
        )
        
        # Initialize database if enabled
        self.db = None
        if config.get('database', {}).get('enabled', False):
            db_path = config['database'].get('path', 'data/inference_results.db')
            self.db = InferenceDB(db_path)
            print(f"Database enabled: {db_path}")
        
        print("Pipeline initialized")
    
    def process_image(self, image_path: str, depth_image_path: str = None, 
                     save_results: bool = True) -> dict:
        """
        Process single image through full pipeline
        
        Args:
            image_path: Path to depth image
            depth_image_path: Path to depth map (optional, if separate from RGB)
            save_results: Whether to save results to database
            
        Returns:
            Dict with results:
                - 'image_path': input image path
                - 'instances': list of instance predictions
                - 'processing_time': total time (seconds)
        """
        start_time = time.time()
        
        # Stage 1: Segmentation
        print(f"Segmenting image: {image_path}")
        instances_seg = self.segmentation.segment_image(image_path)
        print(f"Found {len(instances_seg)} instances")
        
        # Stage 2 & 3: Features + Weight prediction for each instance
        instances_results = []
        for inst in instances_seg:
            # Extract features
            features = self.feature_extractor.extract_all_features(
                mask=inst['mask'],
                maskImg=inst['maskImg'],
                depth_image=None  # TODO: Load depth image if provided
            )
            
            # Predict weight
            predicted_weight = self.weight_predictor.predict(features)
            
            instances_results.append({
                'instance_id': inst['instance_id'],
                'predicted_weight': predicted_weight,
                'confidence_score': inst['score'],
                'box': inst['box'],
                'mask': inst['mask'],
                'maskImg': inst['maskImg'],
                'features': features.tolist()  # For debugging/storage
            })
        
        processing_time = time.time() - start_time
        
        results = {
            'image_path': image_path,
            'instances': instances_results,
            'processing_time': processing_time,
            'num_instances': len(instances_results)
        }
        
        # Save to database if enabled
        if self.db and save_results:
            session_id = self.db.save_inference_session(
                image_path=image_path,
                instances=instances_results,
                processing_time=processing_time,
                config=self.config
            )
            results['session_id'] = session_id
        
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
    """Load configuration from file or use defaults"""
    # Get deployment directory
    deployment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve relative paths to absolute paths relative to deployment/
        if 'segmentation' in config and 'model_path' in config['segmentation']:
            if not os.path.isabs(config['segmentation']['model_path']):
                config['segmentation']['model_path'] = os.path.join(deployment_dir, config['segmentation']['model_path'])
        
        if 'features' in config and 'resnet_weights_path' in config['features'] and config['features']['resnet_weights_path']:
            if not os.path.isabs(config['features']['resnet_weights_path']):
                config['features']['resnet_weights_path'] = os.path.join(deployment_dir, config['features']['resnet_weights_path'])
        
        if 'gbdt' in config and 'model_path' in config['gbdt']:
            if not os.path.isabs(config['gbdt']['model_path']):
                config['gbdt']['model_path'] = os.path.join(deployment_dir, config['gbdt']['model_path'])
        
        if 'database' in config and 'path' in config['database']:
            if not os.path.isabs(config['database']['path']):
                config['database']['path'] = os.path.join(deployment_dir, config['database']['path'])
        
        return config
    
    # Default configuration
    return {
        'segmentation': {
            'model_path': os.path.join(deployment_dir, 'models/segmentation/yolo/best_n.pt'),
            'confidence_threshold': 0.90
        },
        'features': {
            'resnet_weights_path': None  # Use ImageNet pretrained if None
        },
        'gbdt': {
            'model_path': os.path.join(deployment_dir, 'models/gbdt/lgbm_data_20210206-1198/2025-11-21_17-33/result.pkl'),
            'model_type': 'lightgbm'
        },
        'database': {
            'enabled': True,
            'path': os.path.join(deployment_dir, 'data/inference_results.db')
        },
        'device': None  # Auto-detect
    }


def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description='MFF-GBDT MVP Inference Pipeline')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path or directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--no-db', action='store_true',
                       help='Disable database storage')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.no_db:
        config['database']['enabled'] = False
    if args.device:
        config['device'] = args.device
    
    # Initialize pipeline
    pipeline = MVPInferencePipeline(config)
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single image
        results = pipeline.process_image(str(input_path))
        print(f"\nResults:")
        print(f"  Image: {results['image_path']}")
        print(f"  Instances: {results['num_instances']}")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        for i, inst in enumerate(results['instances']):
            print(f"  Instance {i+1}: {inst['predicted_weight']:.3f} kg (confidence: {inst['confidence_score']:.2f})")
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


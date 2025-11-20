"""
Main entrypoint for MVP inference pipeline
Orchestrates: Segmentation → Features → GBDT
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.segmentation import SegmentationInference
from inference.feature_extract import FeatureExtractor
from inference.predict_weight import WeightPredictor
from inference.database import InferenceDB


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
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # Default configuration
    return {
        'segmentation': {
            'model_path': 'data/models/segmentation/weight/205-model-73-100.pth',
            'confidence_threshold': 0.90
        },
        'features': {
            'resnet_weights_path': None  # Use ImageNet pretrained if None
        },
        'gbdt': {
            'model_path': 'data/outputs/exps/lgbm_data_20210206-1198/2025-11-20_15-22/result.pkl',
            'model_type': 'lightgbm'
        },
        'database': {
            'enabled': True,
            'path': 'data/inference_results.db'
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


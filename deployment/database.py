"""
Database module for MVP inference
Stores inference results, configurations, and metadata
Supports SQLite (default) with option to extend to PostgreSQL/MySQL
"""

import sys
import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class InferenceDB:
    """Database interface for storing inference results"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file (relative to deployment/ or absolute)
        """
        if db_path is None:
            # Default: deployment/data/inference_results.db
            deployment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(deployment_dir, 'data/inference_results.db')
        elif not os.path.isabs(db_path):
            # Relative path: resolve relative to deployment/
            deployment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(deployment_dir, db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Inference sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                image_path TEXT NOT NULL,
                num_instances INTEGER,
                processing_time REAL,
                config_json TEXT
            )
        """)
        
        # Instance predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS instance_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                instance_id INTEGER NOT NULL,
                predicted_weight REAL NOT NULL,
                confidence_score REAL,
                features_json TEXT,
                box_x1 INTEGER,
                box_y1 INTEGER,
                box_x2 INTEGER,
                box_y2 INTEGER,
                FOREIGN KEY (session_id) REFERENCES inference_sessions(session_id)
            )
        """)
        
        # Model configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_configs (
                config_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                description TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Bird records table (final aggregated results per track)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bird_records (
                record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                cam_id TEXT,
                track_id INTEGER NOT NULL,
                median_weight REAL NOT NULL,
                duration_in_frames INTEGER NOT NULL,
                num_predictions INTEGER NOT NULL
            )
        """)
        
        self.conn.commit()
    
    def save_inference_session(self, image_path: str, instances: List[Dict], 
                              processing_time: float, config: Dict = None) -> int:
        """
        Save inference session results
        
        Args:
            image_path: Path to input image
            instances: List of instance predictions
            processing_time: Time taken for inference (seconds)
            config: Configuration dict (optional)
            
        Returns:
            session_id
        """
        cursor = self.conn.cursor()
        
        # Insert session
        cursor.execute("""
            INSERT INTO inference_sessions 
            (timestamp, image_path, num_instances, processing_time, config_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            image_path,
            len(instances),
            processing_time,
            json.dumps(config) if config else None
        ))
        
        session_id = cursor.lastrowid
        
        # Insert instance predictions
        for instance in instances:
            cursor.execute("""
                INSERT INTO instance_predictions
                (session_id, instance_id, predicted_weight, confidence_score,
                 features_json, box_x1, box_y1, box_x2, box_y2)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                instance.get('instance_id', 0),
                instance.get('predicted_weight', 0.0),
                instance.get('score', 0.0),
                json.dumps(instance.get('features', [])),
                instance.get('box', [0, 0, 0, 0])[0] if instance.get('box') else 0,
                instance.get('box', [0, 0, 0, 0])[1] if instance.get('box') else 0,
                instance.get('box', [0, 0, 0, 0])[2] if instance.get('box') else 0,
                instance.get('box', [0, 0, 0, 0])[3] if instance.get('box') else 0,
            ))
        
        self.conn.commit()
        return session_id
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """Get recent inference sessions"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM inference_sessions
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_session_predictions(self, session_id: int) -> List[Dict]:
        """Get all predictions for a session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM instance_predictions
            WHERE session_id = ?
            ORDER BY instance_id
        """, (session_id,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def save_model_config(self, model_type: str, model_path: str, 
                         description: str = None) -> int:
        """Save model configuration"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_configs (model_type, model_path, created_at, description)
            VALUES (?, ?, ?, ?)
        """, (
            model_type,
            model_path,
            datetime.now().isoformat(),
            description
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_active_model_config(self, model_type: str) -> Optional[Dict]:
        """Get active model configuration"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM model_configs
            WHERE model_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_type,))
        
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def save_bird_record(self, timestamp: str, cam_id: str, track_id: int,
                        median_weight: float, duration_in_frames: int,
                        num_predictions: int) -> int:
        """
        Save aggregated bird record when track is lost.
        
        Args:
            timestamp: Timestamp when track was lost (ISO format string)
            cam_id: Camera identifier
            track_id: Track ID
            median_weight: Median weight of the track (kg)
            duration_in_frames: Duration of track in frames
            num_predictions: Number of weight predictions in track
            
        Returns:
            record_id
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO bird_records
            (timestamp, cam_id, track_id, median_weight, duration_in_frames, num_predictions)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            cam_id,
            track_id,
            median_weight,
            duration_in_frames,
            num_predictions
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_bird_records(self, limit: int = 10) -> List[Dict]:
        """
        Get recent bird records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of bird record dicts
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM bird_records
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection"""
        self.conn.close()


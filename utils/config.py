"""
Configuration settings for the Facial Recognition Attendance System - FaceNet512 Edition

This module contains all configuration settings, constants, and environment variables
used throughout the application with optimized settings for FaceNet512.

Author: AI Assistant
Date: 2024
"""

import os
from datetime import timedelta

class Config:
    """
    Base configuration class containing all application settings.
    
    This class defines default values for database connections, file paths,
    security settings, and application-specific constants optimized for FaceNet512.
    """
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///attendance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # FaceNet512 Face Recognition Configuration (Adaptive based on available libraries)
    # These thresholds are automatically adjusted based on the embedding method used
    
    # FaceNet512 Settings (512-dimensional embeddings - Primary)
    FACENET512_SIMILARITY_THRESHOLD = 0.4   # Optimized for 512D embeddings
    FACENET512_LEARNING_THRESHOLD = 0.5     # Higher quality threshold for learning
    FACENET512_MODEL = "Facenet512"          # DeepFace model name
    
    # Fallback Settings (128-dimensional embeddings - Secondary)
    FACE_SIMILARITY_THRESHOLD = 0.6         # Will be auto-adjusted: 0.4 for FaceNet512, 0.6 for dlib, 0.85 for MediaPipe
    FACE_LEARNING_THRESHOLD = 0.7           # Will be auto-adjusted: 0.5 for FaceNet512, 0.7 for dlib, 0.9 for MediaPipe  
    MAX_FACE_EMBEDDINGS_PER_EMPLOYEE = 50   # Maximum embeddings stored per employee
    
    # Face Quality Thresholds
    MIN_FACE_CONFIDENCE = 0.8  # Minimum face detection confidence (80%)
    MIN_FACE_SIZE_RATIO = 0.02  # Minimum face size relative to image (2%)
    MAX_FACE_SIZE_RATIO = 0.6   # Maximum face size relative to image (60%)
    
    # Legacy Deep Learning Model Configuration (for fallback)
    FACE_RECOGNITION_MODEL = 'large'  # 'small' for speed (faster), 'large' for accuracy (better)
    FACE_RECOGNITION_JITTERS = 1      # Number of re-samples for better accuracy (1-10, higher = more accurate but slower)
    
    # TensorFlow/DeepFace Configuration
    TF_ENABLE_GPU = True  # Enable GPU acceleration if available
    TF_MEMORY_GROWTH = True  # Allow memory growth for TensorFlow
    
    # QR Code Configuration
    QR_CODE_SIZE = 10  # Size of QR codes in pixels
    QR_CODE_VERSION = 1  # QR code version (1-40)
    
    # Camera Configuration
    CAMERA_RESOLUTIONS = {
        'low': (640, 480),
        'medium': (1280, 720),
        'high': (1920, 1080)
    }
    DEFAULT_CAMERA_RESOLUTION = 'medium'
    
    # Attendance Configuration
    ATTENDANCE_SNAPSHOT_COUNT = 5  # Number of snapshots to take for verification
    ATTENDANCE_SNAPSHOT_INTERVAL = 1  # Seconds between snapshots
    
    # Debugging and Testing Configuration
    ALLOW_DUPLICATE_ATTENDANCE = True  # Set to False in production to prevent duplicate entries within 5 minutes
    
    # Face Registration Configuration
    REGISTRATION_SNAPSHOT_COUNT = 50  # Maximum snapshots for registration
    REGISTRATION_INTERVALS = [1, 2, 3]  # Available intervals in seconds
    REGISTRATION_COUNTDOWN = 3  # Countdown before taking pictures
    
    # PDF Report Configuration
    PDF_TITLE = "Daily Time Record"
    PDF_AUTHOR = "Facial Recognition Attendance System"
    PDF_SUBJECT = "Employee Attendance Report"
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'app.log')
    
    # Face Recognition Capture Settings
    FACE_CAPTURE_COUNT_REGISTRATION = 10  # Number of images to capture during employee registration
    FACE_CAPTURE_COUNT_ATTENDANCE = 1     # Number of images to capture during attendance verification
    FACE_CAPTURE_INTERVAL = 1000          # Milliseconds between captures
    
    # Face Recognition Thresholds (FaceNet512 optimized)
    FACE_SIMILARITY_THRESHOLD_FACENET512 = 0.4    # 40% threshold for FaceNet512 (512-dim)
    FACE_SIMILARITY_THRESHOLD_DLIB = 0.6          # 60% threshold for dlib (128-dim) 
    FACE_SIMILARITY_THRESHOLD_MEDIAPIPE = 0.85    # 85% threshold for MediaPipe fallback (1404-dim)
    
    # Learning thresholds (when to add new embeddings)
    FACE_LEARNING_THRESHOLD_FACENET512 = 0.5      # 50% for FaceNet512
    FACE_LEARNING_THRESHOLD_DLIB = 0.7            # 70% for dlib
    FACE_LEARNING_THRESHOLD_MEDIAPIPE = 0.9       # 90% for MediaPipe
    
    # Face Detection Settings
    FACE_DETECTION_CONFIDENCE = 0.8               # Minimum confidence for face detection
    MAX_FACE_EMBEDDINGS_PER_EMPLOYEE = 50         # Maximum stored embeddings per employee
    
    @staticmethod
    def init_app(app):
        """
        Initialize application with configuration.
        
        Args:
            app: Flask application instance
        """
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
        
        # Configure TensorFlow for optimal performance
        try:
            import tensorflow as tf
            
            # Set memory growth
            if Config.TF_MEMORY_GROWTH:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable GPU if available
            if not Config.TF_ENABLE_GPU:
                tf.config.set_visible_devices([], 'GPU')
                
        except ImportError:
            pass  # TensorFlow not available

class DevelopmentConfig(Config):
    """Development configuration with debug enabled."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///attendance_dev.db'

class ProductionConfig(Config):
    """Production configuration with security settings."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    
    # Production optimizations
    TF_MEMORY_GROWTH = True
    TF_ENABLE_GPU = True
    ALLOW_DUPLICATE_ATTENDANCE = False  # Disable duplicates in production
    
    @classmethod
    def init_app(cls, app):
        """Initialize production app with additional security."""
        Config.init_app(app)
        
        # Production-specific configurations
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            file_handler = RotatingFileHandler(
                cls.LOG_FILE, maxBytes=10240000, backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('FaceNet512 Facial Recognition Attendance System startup')

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

# Configuration dictionary for easy access
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 
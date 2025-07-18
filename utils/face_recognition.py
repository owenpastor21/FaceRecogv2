"""
Face Recognition Utility - FaceNet512 Edition

This module provides state-of-the-art face recognition functionality using FaceNet512
for 512-dimensional embeddings with superior discrimination capabilities.

Architecture:
- MediaPipe: Face detection (fast, reliable)
- FaceNet512: Deep learning embeddings (512-dim, highly discriminative)
- Fallback: Original MediaPipe system for compatibility

Author: AI Assistant
Date: 2024
"""

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import base64
import io
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FaceNet512 deep learning face recognition
try:
    from deepface import DeepFace
    import tensorflow as tf
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    FACENET512_AVAILABLE = True
    logger.info("FaceNet512 (DeepFace) library loaded successfully")
except ImportError as e:
    FACENET512_AVAILABLE = False
    logger.warning(f"FaceNet512 library not available: {str(e)}")
    logger.warning("Install with: pip install deepface tensorflow")

# Fallback deep learning face recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("face_recognition library loaded successfully (fallback)")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition library not available (fallback)")

class FaceRecognitionEngine:
    """
    Advanced face recognition engine using FaceNet512 for 512-dimensional embeddings.
    
    This class provides:
    - FaceNet512: State-of-the-art 512-dimensional face embeddings
    - MediaPipe: Fast, reliable face detection
    - Fallback: Previous systems for compatibility
    
    The 512D embeddings provide significantly better discrimination than 128D systems.
    """
    
    def __init__(self):
        """
        Initialize the face recognition engine with FaceNet512 and fallback options.
        """
        # Initialize MediaPipe Face Detection (for fast face localization)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Import config
        from utils.config import Config
        
        # Configure based on available libraries
        if FACENET512_AVAILABLE:
            # Use FaceNet512 embeddings (512-dimensional)
            self.use_facenet512 = True
            self.facenet_model = "Facenet512"  # 512-dimensional model
            self.embedding_dimension = 512  # FaceNet512 embeddings are 512-dim
            
            # Optimized thresholds for FaceNet512 (more discriminative)
            self.similarity_threshold = 0.4   # Lower threshold due to better discrimination
            self.learning_threshold = 0.5     # Higher quality threshold for learning
            
            logger.info("Using FaceNet512 face recognition (512-dimensional embeddings)")
            logger.info(f"Optimized similarity threshold: {self.similarity_threshold}")
            
        elif FACE_RECOGNITION_AVAILABLE:
            # Fallback to dlib ResNet embeddings
            self.use_facenet512 = False
            self.use_deep_learning = True
            self.face_recognition_model = Config.FACE_RECOGNITION_MODEL
            self.face_recognition_jitters = Config.FACE_RECOGNITION_JITTERS
            self.embedding_dimension = 128  # dlib embeddings are 128-dim
            
            # Original thresholds for 128D embeddings
            self.similarity_threshold = 0.6
            self.learning_threshold = 0.7
            
            logger.info("Using dlib ResNet face recognition (128-dimensional embeddings - fallback)")
            logger.info(f"Similarity threshold: {self.similarity_threshold}")
            
        else:
            # Final fallback to MediaPipe landmarks
            self.use_facenet512 = False
            self.use_deep_learning = False
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.embedding_dimension = 1404  # MediaPipe landmarks (468 * 3)
            
            # Higher thresholds for MediaPipe landmarks (less discriminative)
            self.similarity_threshold = 0.85
            self.learning_threshold = 0.9
            
            logger.info("Using MediaPipe landmark embeddings (final fallback)")
            logger.info(f"Adjusted similarity threshold: {self.similarity_threshold}")
        
        logger.info(f"Face Recognition Engine initialized successfully")
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using MediaPipe.
        
        Args:
            image (np.ndarray): Input image as numpy array (BGR format)
            
        Returns:
            List[Dict]: List of detected faces with bounding boxes and confidence scores
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Get confidence score
                confidence = float(detection.score[0])  # Convert to Python float for JSON serialization
                
                faces.append({
                    'bbox': (x, y, width, height),
                    'confidence': confidence,
                    'detection': detection
                })
        
        return faces
    
    def extract_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract face region from image using bounding box.
        
        Args:
            image (np.ndarray): Input image
            bbox (Tuple[int, int, int, int]): Bounding box (x, y, width, height)
            
        Returns:
            np.ndarray: Cropped face region
        """
        x, y, width, height = bbox
        
        # Add padding to include more of the face
        padding = int(min(width, height) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(image.shape[1] - x, width + 2 * padding)
        height = min(image.shape[0] - y, height + 2 * padding)
        
        face_region = image[y:y+height, x:x+width]
        return face_region
    
    def generate_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate face embedding using FaceNet512, dlib ResNet, or MediaPipe landmarks.
        
        Args:
            image (np.ndarray): Input image containing a face (BGR format)
            
        Returns:
            Optional[np.ndarray]: Face embedding vector or None if no face detected
        """
        try:
            if self.use_facenet512:
                return self._generate_facenet512_embedding(image)
            elif hasattr(self, 'use_deep_learning') and self.use_deep_learning:
                return self._generate_deep_learning_embedding(image)
            else:
                return self._generate_improved_mediapipe_embedding(image)
        except Exception as e:
            logger.error(f"Error generating face embedding: {str(e)}")
            return None
    
    def _generate_facenet512_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate FaceNet512 face embedding using DeepFace library.
        
        This method uses Google's FaceNet with 512-dimensional embeddings that provide
        superior discrimination compared to 128-dimensional models.
        
        Args:
            image (np.ndarray): Input image containing a face (BGR format)
            
        Returns:
            Optional[np.ndarray]: 512-dimensional face embedding or None if no face detected
        """
        try:
            # Convert BGR to RGB (DeepFace expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate face embedding using FaceNet512
            # DeepFace.represent returns a list of dictionaries
            embeddings = DeepFace.represent(
                img_path=rgb_image,
                model_name=self.facenet_model,
                enforce_detection=False,  # Allow processing even if face detection is uncertain
                detector_backend='skip',  # Skip internal detection, use our MediaPipe detection
                align=True,               # Enable face alignment for better results
                normalization='base'      # Standard normalization
            )
            
            if not embeddings or len(embeddings) == 0:
                return None
            
            # Get the first face embedding (512-dimensional vector)
            embedding = np.array(embeddings[0]['embedding'], dtype=np.float32)
            
            # Ensure L2 normalization for consistent similarity calculations
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"FaceNet512 embedding failed: {str(e)}. Trying fallback.")
            # Try fallback method if available
            if hasattr(self, 'use_deep_learning') and FACE_RECOGNITION_AVAILABLE:
                return self._generate_deep_learning_embedding(image)
            else:
                return self._generate_improved_mediapipe_embedding(image)
    
    def _generate_deep_learning_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate deep learning face embedding using face_recognition library (fallback).
        
        This method uses dlib's ResNet-based face recognition model to generate
        a 128-dimensional embedding as a fallback option.
        
        Args:
            image (np.ndarray): Input image containing a face (BGR format)
            
        Returns:
            Optional[np.ndarray]: 128-dimensional face embedding or None if no face detected
        """
        try:
            # Convert BGR to RGB (face_recognition expects RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate face embeddings using deep learning
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                model=self.face_recognition_model,
                num_jitters=self.face_recognition_jitters
            )
            
            if len(face_encodings) == 0:
                return None
            
            # Get the first face encoding (128-dimensional vector)
            embedding = face_encodings[0]
            
            # Ensure it's a float32 numpy array
            embedding = np.array(embedding, dtype=np.float32)
            
            # L2 normalization for consistency
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Deep learning embedding failed: {str(e)}. Using MediaPipe fallback.")
            return self._generate_improved_mediapipe_embedding(image)
    
    def _generate_improved_mediapipe_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate improved MediaPipe landmark-based embedding (final fallback).
        
        Args:
            image (np.ndarray): Input image containing a face (BGR format)
            
        Returns:
            Optional[np.ndarray]: Improved MediaPipe embedding or None if no face detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face mesh landmarks
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Improved normalization for better discrimination
        x_coords = landmarks[0::3]
        y_coords = landmarks[1::3]
        z_coords = landmarks[2::3]
        
        # Center coordinates
        center_x, center_y, center_z = np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)
        
        # Calculate face scale
        face_width = np.max(x_coords) - np.min(x_coords)
        face_height = np.max(y_coords) - np.min(y_coords)
        face_scale = max(face_width, face_height)
        
        # Normalize landmarks
        normalized_landmarks = []
        for i in range(0, len(landmarks), 3):
            norm_x = (landmarks[i] - center_x) / (face_scale + 1e-8)
            norm_y = (landmarks[i+1] - center_y) / (face_scale + 1e-8)
            norm_z = (landmarks[i+2] - center_z) / (face_scale + 1e-8)
            normalized_landmarks.extend([norm_x, norm_y, norm_z])
        
        embedding = np.array(normalized_landmarks, dtype=np.float32)
        
        # L2 normalization
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings.
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Ensure value is between 0 and 1 and convert to Python float
        return float(max(0, min(1, similarity)))
    
    def process_image_for_attendance(self, image: np.ndarray) -> Dict:
        """
        Process an image for attendance verification.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Dict: Processing results including face detection and embedding
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        if not faces:
            return {
                'success': False,
                'error': 'No faces detected in image',
                'faces': []
            }
        
        # Process the first detected face
        face = faces[0]
        face_region = self.extract_face_region(image, face['bbox'])
        
        # Generate embedding
        embedding = self.generate_face_embedding(face_region)
        
        if embedding is None:
            return {
                'success': False,
                'error': 'Failed to generate face embedding',
                'faces': faces
            }
        
        return {
            'success': True,
            'face_region': face_region,
            'embedding': embedding,
            'confidence': face['confidence'],
            'bbox': face['bbox'],
            'faces': faces
        }
    
    def compare_with_stored_embeddings(self, new_embedding: np.ndarray, 
                                     stored_embeddings: List[np.ndarray]) -> Dict:
        """
        Compare a new face embedding with stored embeddings.
        
        Args:
            new_embedding (np.ndarray): New face embedding to compare
            stored_embeddings (List[np.ndarray]): List of stored embeddings
            
        Returns:
            Dict: Comparison results including similarity scores and average
        """
        if not stored_embeddings:
            return {
                'success': False,
                'error': 'No stored embeddings to compare with',
                'similarities': [],
                'average_similarity': 0.0,
                'max_similarity': 0.0
            }
        
        # Calculate similarities with all stored embeddings
        similarities = []
        for stored_embedding in stored_embeddings:
            similarity = self.calculate_similarity(new_embedding, stored_embedding)
            similarities.append(float(similarity))  # Convert to Python float for JSON serialization
        
        # Calculate statistics and convert to Python floats
        average_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))
        min_similarity = float(np.min(similarities))
        std_similarity = float(np.std(similarities))
        
        return {
            'success': True,
            'similarities': similarities,
            'average_similarity': average_similarity,
            'max_similarity': max_similarity,
            'min_similarity': min_similarity,
            'std_similarity': std_similarity
        }
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy image array to base64 string.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            str: Base64 encoded image string
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """
        Convert base64 string to numpy image array.
        
        Args:
            base64_string (str): Base64 encoded image string
            
        Returns:
            np.ndarray: Image as numpy array
        """
        # Decode base64
        img_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(img_data))
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        # Convert RGB to BGR if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def validate_face_quality(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
        """
        Validate the quality of a detected face.
        
        Args:
            image (np.ndarray): Input image
            bbox (Tuple[int, int, int, int]): Face bounding box
            
        Returns:
            Dict: Quality assessment results
        """
        x, y, width, height = bbox
        
        # Calculate face size relative to image
        image_area = image.shape[0] * image.shape[1]
        face_area = width * height
        face_ratio = face_area / image_area
        
        # Calculate brightness
        face_region = self.extract_face_region(image, bbox)
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)
        
        # Calculate blur (using Laplacian variance)
        blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Quality assessment
        quality_score = 0.0
        quality_issues = []
        
        # Check face size
        if face_ratio < 0.01:
            quality_issues.append("Face too small")
        elif face_ratio > 0.5:
            quality_issues.append("Face too large")
        else:
            quality_score += 0.4
        
        # Check brightness
        if brightness < 50:
            quality_issues.append("Image too dark")
        elif brightness > 200:
            quality_issues.append("Image too bright")
        else:
            quality_score += 0.3
        
        # Check blur
        if blur_score < 100:
            quality_issues.append("Image too blurry")
        else:
            quality_score += 0.3
        
        return {
            'quality_score': quality_score,
            'face_ratio': face_ratio,
            'brightness': brightness,
            'blur_score': blur_score,
            'quality_issues': quality_issues,
            'is_acceptable': quality_score >= 0.7
        }

# Create global face engine instance
face_engine = FaceRecognitionEngine() 
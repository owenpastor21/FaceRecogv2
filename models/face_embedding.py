"""
Face Embedding Model

This module defines the FaceEmbedding database model which stores face embeddings
for each employee, enabling face recognition functionality.

Author: AI Assistant
Date: 2024
"""

from datetime import datetime
import numpy as np
import json

# Import shared db instance from models package
from models import db

class FaceEmbedding(db.Model):
    """
    Face Embedding model representing face embeddings for employees.
    
    This model stores face embeddings as numpy arrays, along with metadata about
    when and how the embedding was created. Each employee can have multiple
    embeddings to improve recognition accuracy.
    """
    
    __tablename__ = 'face_embeddings'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Foreign key to employee
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False, index=True)
    
    # Embedding data
    embedding_data = db.Column(db.Text, nullable=False)  # JSON-encoded numpy array
    embedding_dimension = db.Column(db.Integer, nullable=False, default=512)
    
    # Source information
    source_image_path = db.Column(db.String(255))  # Path to source image
    source_image_data = db.Column(db.Text)  # Base64 encoded source image
    
    # Metadata
    similarity_score = db.Column(db.Float)  # Score when this embedding was created
    created_from_attendance = db.Column(db.Boolean, default=False)  # Whether created from attendance
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendances.id'))  # Related attendance record
    
    # Quality metrics
    face_confidence = db.Column(db.Float)  # Confidence in face detection
    face_quality_score = db.Column(db.Float)  # Overall face quality score
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, employee_id, embedding, source_image_data=None, similarity_score=None,
                 created_from_attendance=False, attendance_id=None, face_confidence=None,
                 face_quality_score=None, source_image_path=None):
        """
        Initialize a new FaceEmbedding instance.
        
        Args:
            employee_id (int): ID of the employee
            embedding (numpy.ndarray): Face embedding vector
            source_image_data (str, optional): Base64 encoded source image
            similarity_score (float, optional): Similarity score when created
            created_from_attendance (bool): Whether created from attendance
            attendance_id (int, optional): Related attendance record ID
            face_confidence (float, optional): Face detection confidence
            face_quality_score (float, optional): Face quality score
            source_image_path (str, optional): Path to source image file
        """
        self.employee_id = employee_id
        self.embedding_data = self._encode_embedding(embedding)
        self.embedding_dimension = len(embedding)
        self.source_image_data = source_image_data
        self.similarity_score = similarity_score
        self.created_from_attendance = created_from_attendance
        self.attendance_id = attendance_id
        self.face_confidence = face_confidence
        self.face_quality_score = face_quality_score
        self.source_image_path = source_image_path
    
    def _encode_embedding(self, embedding):
        """
        Encode numpy array to JSON string for database storage.
        
        Args:
            embedding (numpy.ndarray): Face embedding vector
            
        Returns:
            str: JSON-encoded embedding data
        """
        return json.dumps(embedding.tolist())
    
    def _decode_embedding(self):
        """
        Decode JSON string to numpy array.
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        data = json.loads(self.embedding_data)
        return np.array(data, dtype=np.float32)
    
    @property
    def embedding(self):
        """
        Get the face embedding as a numpy array.
        
        Returns:
            numpy.ndarray: Face embedding vector
        """
        return self._decode_embedding()
    
    def get_source_image_url(self):
        """
        Get the URL for the source image.
        
        Returns:
            str: URL to the source image
        """
        if self.source_image_path:
            return f"/uploads/embeddings/{self.source_image_path}"
        return None
    
    def save_source_image(self, image_data, filename=None):
        """
        Save the source image to disk.
        
        Args:
            image_data (bytes): Image data to save
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to saved file
        """
        import os
        from utils.config import Config
        
        if not filename:
            timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
            filename = f"emp_{self.employee_id}_embed_{self.id}_{timestamp}.jpg"
        
        embedding_dir = os.path.join(Config.UPLOAD_FOLDER, 'embeddings')
        os.makedirs(embedding_dir, exist_ok=True)
        
        file_path = os.path.join(embedding_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        self.source_image_path = filename
        return file_path
    
    def calculate_similarity(self, other_embedding):
        """
        Calculate cosine similarity with another embedding.
        
        Args:
            other_embedding (numpy.ndarray): Other face embedding
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        from scipy.spatial.distance import cosine
        
        embedding1 = self.embedding
        embedding2 = other_embedding
        
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1_norm, embedding2_norm)
        return max(0, min(1, similarity))  # Ensure value is between 0 and 1
    
    def to_dict(self):
        """
        Convert face embedding to dictionary representation.
        
        Returns:
            dict: Face embedding data as dictionary
        """
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'embedding_dimension': self.embedding_dimension,
            'source_image_url': self.get_source_image_url(),
            'similarity_score': self.similarity_score,
            'created_from_attendance': self.created_from_attendance,
            'attendance_id': self.attendance_id,
            'face_confidence': self.face_confidence,
            'face_quality_score': self.face_quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_employee_embeddings(cls, employee_id):
        """
        Get all face embeddings for an employee.
        
        Args:
            employee_id (int): Employee ID
            
        Returns:
            list: List of face embeddings
        """
        return cls.query.filter_by(employee_id=employee_id).order_by(cls.created_at.desc()).all()
    
    @classmethod
    def get_employee_embeddings_as_arrays(cls, employee_id):
        """
        Get all face embeddings for an employee as numpy arrays.
        
        Args:
            employee_id (int): Employee ID
            
        Returns:
            list: List of numpy arrays representing face embeddings
        """
        embeddings = cls.get_employee_embeddings(employee_id)
        return [emb.embedding for emb in embeddings]
    
    @classmethod
    def delete_oldest_embedding(cls, employee_id):
        """
        Delete the oldest face embedding for an employee.
        
        Args:
            employee_id (int): Employee ID
            
        Returns:
            bool: True if deleted, False if no embeddings found
        """
        oldest = cls.query.filter_by(employee_id=employee_id).order_by(cls.created_at.asc()).first()
        if oldest:
            db.session.delete(oldest)
            db.session.commit()
            return True
        return False
    
    def __repr__(self):
        """String representation of the FaceEmbedding."""
        return f'<FaceEmbedding {self.id}: Employee {self.employee_id}, Dim {self.embedding_dimension}>' 
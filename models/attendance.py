"""
Attendance Model

This module defines the Attendance database model which tracks employee time in/out
records, including face verification data and snapshots.

Author: AI Assistant
Date: 2024
"""

from datetime import datetime
import json

# Import shared db instance from models package
from models import db

class Attendance(db.Model):
    """
    Attendance model representing employee time in/out records.
    
    This model tracks when employees check in and out, including face verification
    scores, snapshots, and metadata about the attendance event.
    """
    
    __tablename__ = 'attendances'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Foreign key to employee
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False, index=True)
    
    # Attendance details
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now, index=True)
    type = db.Column(db.String(10), nullable=False)  # 'in' or 'out'
    
    # Face verification data
    face_similarity_score = db.Column(db.Float, nullable=False)
    face_snapshot_path = db.Column(db.String(255))  # Path to stored snapshot image
    face_snapshot_data = db.Column(db.Text)  # Base64 encoded snapshot (for immediate access)
    
    # Verification metadata
    verification_method = db.Column(db.String(20), default='face_qr')  # 'face_qr', 'face_only', 'qr_only'
    qr_code_scanned = db.Column(db.Boolean, default=True)
    face_verified = db.Column(db.Boolean, default=True)
    
    # Additional metadata
    device_info = db.Column(db.String(255))  # Device used for attendance
    location_info = db.Column(db.String(255))  # Location information if available
    notes = db.Column(db.Text)  # Additional notes
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __init__(self, employee_id, type, face_similarity_score, face_snapshot_data=None,
                 verification_method='face_qr', device_info=None, location_info=None, notes=None):
        """
        Initialize a new Attendance instance.
        
        Args:
            employee_id (int): ID of the employee
            type (str): Type of attendance ('in' or 'out')
            face_similarity_score (float): Face verification similarity score
            face_snapshot_data (str, optional): Base64 encoded face snapshot
            verification_method (str): Method used for verification
            device_info (str, optional): Information about the device used
            location_info (str, optional): Location information
            notes (str, optional): Additional notes
        """
        self.employee_id = employee_id
        self.type = type.lower()
        self.face_similarity_score = face_similarity_score
        self.face_snapshot_data = face_snapshot_data
        self.verification_method = verification_method
        self.device_info = device_info
        self.location_info = location_info
        self.notes = notes
        
        # Set verification flags based on method
        self.qr_code_scanned = 'qr' in verification_method
        self.face_verified = 'face' in verification_method
    
    @property
    def is_time_in(self):
        """
        Check if this is a time in record.
        
        Returns:
            bool: True if time in, False if time out
        """
        return self.type == 'in'
    
    @property
    def is_time_out(self):
        """
        Check if this is a time out record.
        
        Returns:
            bool: True if time out, False if time in
        """
        return self.type == 'out'
    
    @property
    def date(self):
        """
        Get the date of the attendance record.
        
        Returns:
            datetime.date: Date of the attendance
        """
        return self.timestamp.date()
    
    @property
    def time(self):
        """
        Get the time of the attendance record.
        
        Returns:
            datetime.time: Time of the attendance
        """
        return self.timestamp.time()
    
    @property
    def verification_status(self):
        """
        Get the verification status as a string.
        
        Returns:
            str: Verification status description
        """
        if self.face_verified and self.qr_code_scanned:
            return "Face + QR Verified"
        elif self.face_verified:
            return "Face Verified"
        elif self.qr_code_scanned:
            return "QR Verified"
        else:
            return "Not Verified"
    
    def is_verified(self):
        """
        Check if the attendance is properly verified.
        
        Returns:
            bool: True if verified, False otherwise
        """
        from utils.config import Config
        return (self.face_verified and 
                self.qr_code_scanned and 
                self.face_similarity_score >= Config.FACE_SIMILARITY_THRESHOLD)
    
    def get_face_snapshot_url(self):
        """
        Get the URL for the face snapshot image.
        
        Returns:
            str: URL to the face snapshot image
        """
        if self.face_snapshot_path:
            return f"/uploads/snapshots/{self.face_snapshot_path}"
        return None
    
    def to_dict(self):
        """
        Convert attendance to dictionary representation.
        
        Returns:
            dict: Attendance data as dictionary
        """
        return {
            'id': self.id,
            'employee_id': self.employee_id,
            'employee_name': self.employee.full_name if self.employee else None,
            'employee_number': self.employee.employee_number if self.employee else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'type': self.type,
            'date': self.date.isoformat() if self.date else None,
            'time': self.time.isoformat() if self.time else None,
            'face_similarity_score': self.face_similarity_score,
            'face_snapshot_url': self.get_face_snapshot_url(),
            'verification_method': self.verification_method,
            'verification_status': self.verification_status,
            'is_verified': self.is_verified(),
            'qr_code_scanned': self.qr_code_scanned,
            'face_verified': self.face_verified,
            'device_info': self.device_info,
            'location_info': self.location_info,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def get_employee_attendance_by_date_range(cls, employee_id, start_date, end_date):
        """
        Get attendance records for an employee within a date range.
        
        Args:
            employee_id (int): Employee ID
            start_date (datetime.date): Start date
            end_date (datetime.date): End date
            
        Returns:
            list: List of attendance records
        """
        return cls.query.filter(
            cls.employee_id == employee_id,
            db.func.date(cls.timestamp) >= start_date,
            db.func.date(cls.timestamp) <= end_date
        ).order_by(cls.timestamp).all()
    
    @classmethod
    def get_daily_summary(cls, date):
        """
        Get daily attendance summary for all employees.
        
        Args:
            date (datetime.date): Date to get summary for
            
        Returns:
            list: List of daily attendance summaries
        """
        from sqlalchemy import func
        
        return db.session.query(
            cls.employee_id,
            func.min(cls.timestamp).label('first_in'),
            func.max(cls.timestamp).label('last_out'),
            func.count(cls.id).label('total_records')
        ).filter(
            db.func.date(cls.timestamp) == date
        ).group_by(cls.employee_id).all()
    
    def __repr__(self):
        """String representation of the Attendance."""
        return f'<Attendance {self.employee_id}: {self.type} at {self.timestamp}>' 
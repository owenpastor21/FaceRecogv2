"""
Employee Model

This module defines the Employee database model which stores employee information
including personal details, face embeddings, and QR code data.

Author: AI Assistant
Date: 2024
"""

from datetime import datetime
import qrcode
import io
import base64
from PIL import Image

# Import shared db instance from models package
from models import db

class Employee(db.Model):
    """
    Employee model representing an employee in the attendance system.
    
    This model stores employee personal information, generates QR codes for
    identification, and maintains relationships with attendance records and
    face embeddings.
    """
    
    __tablename__ = 'employees'
    
    # Primary key
    id = db.Column(db.Integer, primary_key=True)
    
    # Employee identification
    employee_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    qr_code_data = db.Column(db.Text, unique=True, nullable=False)
    
    # Personal information
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    middle_name = db.Column(db.String(50))
    department = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(20))
    email = db.Column(db.String(100))
    
    # Status and metadata
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    attendances = db.relationship('Attendance', backref='employee', lazy='dynamic', cascade='all, delete-orphan')
    face_embeddings = db.relationship('FaceEmbedding', backref='employee', lazy='dynamic', cascade='all, delete-orphan')
    
    def __init__(self, employee_number, first_name, last_name, department, position, 
                 middle_name=None, contact_number=None, email=None):
        """
        Initialize a new Employee instance.
        
        Args:
            employee_number (str): Unique employee identification number
            first_name (str): Employee's first name
            last_name (str): Employee's last name
            department (str): Employee's department
            position (str): Employee's position/designation
            middle_name (str, optional): Employee's middle name
            contact_number (str, optional): Employee's contact number
            email (str, optional): Employee's email address
        """
        self.employee_number = employee_number
        self.first_name = first_name
        self.last_name = last_name
        self.middle_name = middle_name
        self.department = department
        self.position = position
        self.contact_number = contact_number
        self.email = email
        self.qr_code_data = self._generate_qr_code_data()
    
    def _generate_qr_code_data(self):
        """
        Generate QR code data for the employee.
        
        Returns:
            str: QR code data containing employee number
        """
        return f"EMP_{self.employee_number}"
    
    def generate_qr_code_image(self, size=10):
        """
        Generate QR code image for the employee.
        
        Args:
            size (int): Size of the QR code in pixels
            
        Returns:
            str: Base64 encoded QR code image
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=size,
            border=4,
        )
        qr.add_data(self.qr_code_data)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    @property
    def full_name(self):
        """
        Get the employee's full name.
        
        Returns:
            str: Employee's full name
        """
        if self.middle_name:
            return f"{self.first_name} {self.middle_name} {self.last_name}"
        return f"{self.first_name} {self.last_name}"
    
    @property
    def face_embedding_count(self):
        """
        Get the number of face embeddings for this employee.
        
        Returns:
            int: Number of face embeddings
        """
        return self.face_embeddings.count()
    
    def can_add_face_embedding(self):
        """
        Check if a new face embedding can be added.
        
        Returns:
            bool: True if embedding can be added, False otherwise
        """
        from utils.config import Config
        return self.face_embedding_count < Config.MAX_FACE_EMBEDDINGS_PER_EMPLOYEE
    
    def get_latest_attendance(self):
        """
        Get the employee's most recent attendance record.
        
        Returns:
            Attendance: Latest attendance record or None
        """
        return self.attendances.order_by(db.desc('timestamp')).first()
    
    def get_attendance_by_date(self, date):
        """
        Get attendance records for a specific date.
        
        Args:
            date (datetime.date): Date to get attendance for
            
        Returns:
            list: List of attendance records for the date
        """
        return self.attendances.filter(
            db.func.date(db.text('timestamp')) == date
        ).order_by(db.text('timestamp')).all()
    
    def to_dict(self):
        """
        Convert employee to dictionary representation.
        
        Returns:
            dict: Employee data as dictionary
        """
        return {
            'id': self.id,
            'employee_number': self.employee_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'middle_name': self.middle_name,
            'full_name': self.full_name,
            'department': self.department,
            'position': self.position,
            'contact_number': self.contact_number,
            'email': self.email,
            'is_active': self.is_active,
            'face_embedding_count': self.face_embedding_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        """String representation of the Employee."""
        return f'<Employee {self.employee_number}: {self.full_name}>' 
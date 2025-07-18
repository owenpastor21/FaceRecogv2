"""
Database Models Package

This package contains all database models for the facial recognition attendance system.
"""

from flask_sqlalchemy import SQLAlchemy

# Create shared database instance
db = SQLAlchemy()

# Import all models after db is created
from .employee import Employee
from .attendance import Attendance
from .face_embedding import FaceEmbedding

__all__ = ['db', 'Employee', 'Attendance', 'FaceEmbedding'] 
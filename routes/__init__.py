"""
Routes Package

This package contains all route blueprints for the facial recognition attendance system.
"""

from .attendance import attendance_bp
from .employee import employee_bp
from .admin import admin_bp
from .api import api_bp

__all__ = ['attendance_bp', 'employee_bp', 'admin_bp', 'api_bp'] 
"""
Employee Routes

This module contains routes for employee management including registration,
face registration, and employee data management.

Author: AI Assistant
Date: 2024
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from datetime import datetime
import cv2
import numpy as np
import base64
import logging
import os

# Import models and utilities
from models.employee import Employee
from models.face_embedding import FaceEmbedding
from utils.face_recognition import face_engine
from utils.database import db, commit_changes
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
employee_bp = Blueprint('employee', __name__)

@employee_bp.route('/register')
def register_employee():
    """
    Employee registration page.
    
    Returns:
        str: Rendered employee registration page
    """
    return render_template('employee/register.html')

@employee_bp.route('/face-registration')
def face_registration():
    """
    Face registration page.
    
    Returns:
        str: Rendered face registration page
    """
    return render_template('employee/face_registration.html')

@employee_bp.route('/list')
def list_employees():
    """
    Employee list page.
    
    Returns:
        str: Rendered employee list page
    """
    return render_template('employee/list.html')

@employee_bp.route('/api/register', methods=['POST'])
def api_register_employee():
    """
    Register a new employee.
    
    Expected JSON payload:
    {
        "employee_number": "EMP001",
        "first_name": "John",
        "last_name": "Doe",
        "middle_name": "Smith",
        "department": "IT",
        "position": "Developer",
        "contact_number": "1234567890",
        "email": "john.doe@company.com"
    }
    
    Returns:
        JSON: Registration results
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['employee_number', 'first_name', 'last_name', 'department', 'position']
        for field in required_fields:
            if not data or field not in data or not data[field]:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Check if employee number already exists
        existing_employee = Employee.query.filter_by(employee_number=data['employee_number']).first()
        if existing_employee:
            return jsonify({
                'success': False,
                'error': f'Employee with number {data["employee_number"]} already exists'
            }), 400
        
        # Create new employee
        employee = Employee(
            employee_number=data['employee_number'],
            first_name=data['first_name'],
            last_name=data['last_name'],
            middle_name=data.get('middle_name'),
            department=data['department'],
            position=data['position'],
            contact_number=data.get('contact_number'),
            email=data.get('email')
        )
        
        # Save to database
        db.session.add(employee)
        commit_changes()
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict(),
            'message': f'Employee {employee.full_name} registered successfully'
        })
        
    except Exception as e:
        logger.error(f"Error registering employee: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@employee_bp.route('/api/register-face', methods=['POST'])
def api_register_face():
    """
    Register face embeddings for an employee.
    
    Expected JSON payload:
    {
        "employee_id": 1,
        "images": ["base64_encoded_image1", "base64_encoded_image2", ...],
        "snapshot_count": 50,
        "interval": 2
    }
    
    Returns:
        JSON: Face registration results
    """
    try:
        data = request.get_json()
        
        if not data or 'employee_id' not in data or 'images' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: employee_id and images'
            }), 400
        
        # Get employee
        employee = Employee.query.get(data['employee_id'])
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Check if employee can add more embeddings
        if not employee.can_add_face_embedding():
            return jsonify({
                'success': False,
                'error': f'Employee already has maximum number of face embeddings ({Config.MAX_FACE_EMBEDDINGS_PER_EMPLOYEE})'
            }), 400
        
        # Process images
        images = data['images']
        snapshot_count = data.get('snapshot_count', Config.REGISTRATION_SNAPSHOT_COUNT)
        
        if len(images) < snapshot_count:
            return jsonify({
                'success': False,
                'error': f'Insufficient images. Expected {snapshot_count}, got {len(images)}'
            }), 400
        
        # Process each image and create embeddings
        successful_embeddings = 0
        failed_images = 0
        
        for i, image_data in enumerate(images[:snapshot_count]):
            try:
                # Convert base64 to image
                image = face_engine.base64_to_image(image_data)
                
                # Process image for face recognition
                face_result = face_engine.process_image_for_attendance(image)
                
                if not face_result['success']:
                    logger.warning(f"Failed to process image {i}: {face_result['error']}")
                    failed_images += 1
                    continue
                
                # Validate face quality
                quality_result = face_engine.validate_face_quality(image, face_result['bbox'])
                
                if not quality_result['is_acceptable']:
                    logger.warning(f"Image {i} quality too low: {quality_result['quality_issues']}")
                    failed_images += 1
                    continue
                
                # Check if we can add more embeddings
                if not employee.can_add_face_embedding():
                    logger.info(f"Maximum embeddings reached for employee {employee.id}")
                    break
                
                # Create face embedding
                face_embedding = FaceEmbedding(
                    employee_id=employee.id,
                    embedding=face_result['embedding'],
                    source_image_data=image_data,
                    face_confidence=face_result['confidence'],
                    face_quality_score=quality_result['quality_score']
                )
                
                db.session.add(face_embedding)
                successful_embeddings += 1
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                failed_images += 1
                continue
        
        # Commit all embeddings
        commit_changes()
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict(),
            'successful_embeddings': successful_embeddings,
            'failed_images': failed_images,
            'total_embeddings': employee.face_embedding_count,
            'message': f'Successfully registered {successful_embeddings} face embeddings for {employee.full_name}'
        })
        
    except Exception as e:
        logger.error(f"Error registering face: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@employee_bp.route('/api/upload-face-images', methods=['POST'])
def api_upload_face_images():
    """
    Upload face images for registration.
    
    Expected form data:
    - employee_id: Employee ID
    - images: Multiple image files
    
    Returns:
        JSON: Upload results
    """
    try:
        # Check if employee_id is provided
        employee_id = request.form.get('employee_id')
        if not employee_id:
            return jsonify({
                'success': False,
                'error': 'Missing employee_id'
            }), 400
        
        # Get employee
        employee = Employee.query.get(employee_id)
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Check uploaded files
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No images uploaded'
            }), 400
        
        files = request.files.getlist('images')
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                'success': False,
                'error': 'No valid images selected'
            }), 400
        
        # Process uploaded images
        successful_embeddings = 0
        failed_images = 0
        base64_images = []
        
        for file in files:
            try:
                # Check file extension
                if not _allowed_file(file.filename):
                    failed_images += 1
                    continue
                
                # Read image data
                image_data = file.read()
                
                # Convert to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    failed_images += 1
                    continue
                
                # Process image for face recognition
                face_result = face_engine.process_image_for_attendance(image)
                
                if not face_result['success']:
                    failed_images += 1
                    continue
                
                # Validate face quality
                quality_result = face_engine.validate_face_quality(image, face_result['bbox'])
                
                if not quality_result['is_acceptable']:
                    failed_images += 1
                    continue
                
                # Convert to base64
                base64_image = face_engine.image_to_base64(image)
                base64_images.append(base64_image)
                
                # Check if we can add more embeddings
                if not employee.can_add_face_embedding():
                    break
                
                # Create face embedding
                face_embedding = FaceEmbedding(
                    employee_id=employee.id,
                    embedding=face_result['embedding'],
                    source_image_data=base64_image,
                    face_confidence=face_result['confidence'],
                    face_quality_score=quality_result['quality_score']
                )
                
                db.session.add(face_embedding)
                successful_embeddings += 1
                
            except Exception as e:
                logger.error(f"Error processing uploaded image: {str(e)}")
                failed_images += 1
                continue
        
        # Commit all embeddings
        commit_changes()
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict(),
            'successful_embeddings': successful_embeddings,
            'failed_images': failed_images,
            'total_embeddings': employee.face_embedding_count,
            'base64_images': base64_images,
            'message': f'Successfully processed {successful_embeddings} images for {employee.full_name}'
        })
        
    except Exception as e:
        logger.error(f"Error uploading face images: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@employee_bp.route('/api/list')
def api_list_employees():
    """
    Get list of all employees.
    
    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20)
    - search: Search term for employee name or number
    - department: Filter by department
    - active: Filter by active status (true/false)
    
    Returns:
        JSON: List of employees with pagination
    """
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '')
        department = request.args.get('department', '')
        active = request.args.get('active', '')
        
        # Build query
        query = Employee.query
        
        # Apply filters
        if search:
            search_term = f'%{search}%'
            query = query.filter(
                db.or_(
                    Employee.first_name.ilike(search_term),
                    Employee.last_name.ilike(search_term),
                    Employee.employee_number.ilike(search_term)
                )
            )
        
        if department:
            query = query.filter(Employee.department == department)
        
        if active.lower() in ['true', 'false']:
            is_active = active.lower() == 'true'
            query = query.filter(Employee.is_active == is_active)
        
        # Get paginated results
        pagination = query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        employees = pagination.items
        
        return jsonify({
            'success': True,
            'employees': [emp.to_dict() for emp in employees],
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing employees: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@employee_bp.route('/api/<int:employee_id>')
def api_get_employee(employee_id):
    """
    Get employee details by ID.
    
    Args:
        employee_id (int): Employee ID
        
    Returns:
        JSON: Employee details
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@employee_bp.route('/api/<int:employee_id>/update', methods=['PUT'])
def api_update_employee(employee_id):
    """
    Update employee information.
    
    Args:
        employee_id (int): Employee ID
        
    Expected JSON payload:
    {
        "first_name": "John",
        "last_name": "Doe",
        "middle_name": "Smith",
        "department": "IT",
        "position": "Developer",
        "contact_number": "1234567890",
        "email": "john.doe@company.com",
        "is_active": true
    }
    
    Returns:
        JSON: Update results
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        data = request.get_json()
        
        # Update fields
        if 'first_name' in data:
            employee.first_name = data['first_name']
        if 'last_name' in data:
            employee.last_name = data['last_name']
        if 'middle_name' in data:
            employee.middle_name = data['middle_name']
        if 'department' in data:
            employee.department = data['department']
        if 'position' in data:
            employee.position = data['position']
        if 'contact_number' in data:
            employee.contact_number = data['contact_number']
        if 'email' in data:
            employee.email = data['email']
        if 'is_active' in data:
            employee.is_active = data['is_active']
        
        # Update timestamp
        employee.updated_at = datetime.utcnow()
        
        commit_changes()
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict(),
            'message': f'Employee {employee.full_name} updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@employee_bp.route('/api/<int:employee_id>/delete', methods=['DELETE'])
def api_delete_employee(employee_id):
    """
    Delete an employee.
    
    Args:
        employee_id (int): Employee ID
        
    Returns:
        JSON: Deletion results
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Store employee name for response
        employee_name = employee.full_name
        
        # Delete employee (cascade will delete related records)
        db.session.delete(employee)
        commit_changes()
        
        return jsonify({
            'success': True,
            'message': f'Employee {employee_name} deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

def _allowed_file(filename):
    """
    Check if file extension is allowed.
    
    Args:
        filename (str): File name
        
    Returns:
        bool: True if allowed, False otherwise
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS 
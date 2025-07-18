"""
API Routes

This module contains RESTful API endpoints for the facial recognition attendance system.
These endpoints are designed to be consumed by frontend applications and mobile clients.

Author: AI Assistant
Date: 2024
"""

from flask import Blueprint, request, jsonify, current_app
from datetime import datetime, timedelta
import logging

# Import models and utilities
from models.employee import Employee
from models.attendance import Attendance
from models.face_embedding import FaceEmbedding
from utils.database import db, get_database_stats
from utils.config import Config
from utils.qr_code import qr_manager
from utils.face_recognition import face_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# API Version
API_VERSION = 'v1'

@api_bp.route('/version')
def api_version():
    """
    Get API version information.
    
    Returns:
        JSON: API version and status
    """
    return jsonify({
        'version': API_VERSION,
        'status': 'active',
        'timestamp': datetime.utcnow().isoformat()
    })

# QR Code API Endpoints

@api_bp.route('/employees/<int:employee_id>/qr-code', methods=['GET'])
def get_employee_qr_code(employee_id):
    """
    Generate QR code for an employee.
    
    Args:
        employee_id (int): Employee ID
        
    Query parameters:
    - size: QR code size (default: 10)
    - format: Response format ('json' or 'image', default: 'json')
    
    Returns:
        JSON: QR code data or image
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Get parameters
        size = request.args.get('size', 10, type=int)
        response_format = request.args.get('format', 'json')
        
        # Generate QR code using employee's method
        qr_code_base64 = employee.generate_qr_code_image(size)
        
        if not qr_code_base64:
            return jsonify({
                'success': False,
                'error': 'Failed to generate QR code'
            }), 500
        
        if response_format == 'image':
            # Return raw image data
            from flask import Response
            import base64
            image_data = base64.b64decode(qr_code_base64)
            return Response(image_data, mimetype='image/png')
        else:
            # Return JSON response
            return jsonify({
                'success': True,
                'data': {
                    'employee_id': employee_id,
                    'employee_number': employee.employee_number,
                    'qr_code_data': employee.qr_code_data,
                    'qr_code_image': qr_code_base64,
                    'generated_at': datetime.utcnow().isoformat()
                }
            })
        
    except Exception as e:
        logger.error(f"Error generating QR code for employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>/qr-code/regenerate', methods=['POST'])
def regenerate_employee_qr_code(employee_id):
    """
    Regenerate QR code for an employee.
    
    Args:
        employee_id (int): Employee ID
        
    Returns:
        JSON: New QR code data
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Update QR code data (regenerate with timestamp for uniqueness)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        new_qr_data = f"EMP_{employee.employee_number}_{timestamp}"
        employee.qr_code_data = new_qr_data
        
        # Generate new QR code image using employee's method
        qr_code_base64 = employee.generate_qr_code_image()
        
        if not qr_code_base64:
            return jsonify({
                'success': False,
                'error': 'Failed to regenerate QR code'
            }), 500
        
        # Save to database
        employee.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': {
                'employee_id': employee_id,
                'employee_number': employee.employee_number,
                'qr_code_data': new_qr_data,
                'qr_code_image': qr_code_base64,
                'regenerated_at': datetime.utcnow().isoformat()
            },
            'message': 'QR code regenerated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error regenerating QR code for employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/qr-code/validate', methods=['POST'])
def validate_qr_code():
    """
    Validate QR code data.
    
    Expected JSON payload:
    {
        "qr_data": "EMP_12345",
        "image": "base64_encoded_image" (optional)
    }
    
    Returns:
        JSON: Validation results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Missing request data'
            }), 400
        
        qr_data = data.get('qr_data')
        image_data = data.get('image')
        
        # If image is provided, scan QR code from image
        if image_data and not qr_data:
            qr_result = qr_manager.scan_qr_code_from_base64(image_data)
            if qr_result:
                qr_data = qr_result[0]['data']
            else:
                return jsonify({
                    'success': False,
                    'error': 'No QR code detected in image'
                }), 400
        
        if not qr_data:
            return jsonify({
                'success': False,
                'error': 'QR code data is required'
            }), 400
        
        # Validate QR code format
        validation = qr_manager.validate_qr_code_data(qr_data)
        
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': validation['error']
            }), 400
        
        # Check if employee exists
        employee_number = validation['employee_number']
        employee = Employee.query.filter_by(employee_number=employee_number).first()
        
        if not employee:
            return jsonify({
                'success': False,
                'error': f'Employee with number {employee_number} not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': {
                'valid': True,
                'employee_number': employee_number,
                'employee': employee.to_dict(),
                'qr_data': qr_data
            }
        })
        
    except Exception as e:
        logger.error(f"Error validating QR code: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>/faces', methods=['POST'])
def add_employee_face(employee_id):
    """
    Add face embedding for an employee.
    
    Args:
        employee_id (int): Employee ID
        
    Expected form data:
    - face_image: Image file or base64 data
    
    Returns:
        JSON: Face embedding creation result
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Check if employee can add more face embeddings
        if not employee.can_add_face_embedding():
            return jsonify({
                'success': False,
                'error': 'Maximum face embeddings reached for this employee'
            }), 400
        
        # Get face image data
        face_image_data = None
        
        if 'face_image' in request.files:
            # File upload
            file = request.files['face_image']
            if file.filename:
                import base64
                face_image_data = base64.b64encode(file.read()).decode('utf-8')
        elif request.get_json() and 'face_image' in request.get_json():
            # Base64 data
            face_image_data = request.get_json()['face_image']
        
        if not face_image_data:
            return jsonify({
                'success': False,
                'error': 'No face image provided'
            }), 400
        
        # Process face image using face recognition engine
        from utils.face_recognition import face_engine
        
        try:
            # Convert base64 to image
            image = face_engine.base64_to_image(face_image_data)
            
            # Process image for face recognition
            face_result = face_engine.process_image_for_attendance(image)
            
            if not face_result['success']:
                return jsonify({
                    'success': False,
                    'error': face_result['error']
                }), 400
            
            # Create face embedding
            face_embedding = FaceEmbedding(
                employee_id=employee_id,
                embedding=face_result['embedding'],
                source_image_data=face_image_data,
                face_confidence=face_result.get('confidence'),
                face_quality_score=face_result.get('quality_score')
            )
            
            db.session.add(face_embedding)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'data': face_embedding.to_dict(),
                'message': 'Face embedding added successfully'
            }), 201
            
        except Exception as e:
            logger.error(f"Error processing face image: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Failed to process face image'
            }), 400
        
    except Exception as e:
        logger.error(f"Error adding face embedding for employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>/faces/<int:face_id>', methods=['DELETE'])
def delete_employee_face(employee_id, face_id):
    """
    Delete a face embedding for an employee.
    
    Args:
        employee_id (int): Employee ID
        face_id (int): Face embedding ID
        
    Returns:
        JSON: Deletion confirmation
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        face_embedding = FaceEmbedding.query.filter_by(
            id=face_id, 
            employee_id=employee_id
        ).first()
        
        if not face_embedding:
            return jsonify({
                'success': False,
                'error': 'Face embedding not found'
            }), 404
        
        db.session.delete(face_embedding)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Face embedding deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Error deleting face embedding {face_id} for employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>/embeddings', methods=['GET'])
def get_employee_embeddings(employee_id):
    """
    Get all face embeddings for an employee.
    
    Args:
        employee_id (int): Employee ID
        
    Returns:
        JSON: List of face embeddings
    """
    try:
        employee = Employee.query.get(employee_id)
        
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        embeddings = FaceEmbedding.get_employee_embeddings(employee_id)
        
        return jsonify({
            'success': True,
            'data': [emb.to_dict() for emb in embeddings],
            'total': len(embeddings)
        })
        
    except Exception as e:
        logger.error(f"Error getting embeddings for employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# Employee API Endpoints

@api_bp.route('/employees', methods=['GET'])
def get_employees():
    """
    Get list of employees with pagination and filtering.
    
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
            'data': [emp.to_dict() for emp in employees],
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
        logger.error(f"Error getting employees: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
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
            'data': employee.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees', methods=['POST'])
def create_employee():
    """
    Create a new employee.
    
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
        JSON: Created employee data
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
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': employee.to_dict(),
            'message': f'Employee {employee.full_name} created successfully'
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating employee: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
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
        JSON: Updated employee data
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
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'data': employee.to_dict(),
            'message': f'Employee {employee.full_name} updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating employee {employee_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/employees/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    """
    Delete an employee.
    
    Args:
        employee_id (int): Employee ID
        
    Returns:
        JSON: Deletion confirmation
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
        db.session.commit()
        
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

# Attendance API Endpoints

@api_bp.route('/attendance', methods=['GET'])
def get_attendance():
    """
    Get attendance records with filtering and pagination.
    
    Query parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20)
    - employee_id: Filter by employee ID
    - start_date: Filter by start date (YYYY-MM-DD)
    - end_date: Filter by end date (YYYY-MM-DD)
    - type: Filter by attendance type (in/out)
    
    Returns:
        JSON: List of attendance records
    """
    try:
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        employee_id = request.args.get('employee_id', type=int)
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        attendance_type = request.args.get('type', '')
        
        # Build query
        query = Attendance.query
        
        # Apply filters
        if employee_id:
            query = query.filter(Attendance.employee_id == employee_id)
        
        if start_date:
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
                query = query.filter(db.func.date(Attendance.timestamp) >= start_dt)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid start_date format. Use YYYY-MM-DD'
                }), 400
        
        if end_date:
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
                query = query.filter(db.func.date(Attendance.timestamp) <= end_dt)
            except ValueError:
                return jsonify({
                    'success': False,
                    'error': 'Invalid end_date format. Use YYYY-MM-DD'
                }), 400
        
        if attendance_type in ['in', 'out']:
            query = query.filter(Attendance.type == attendance_type)
        
        # Get paginated results
        pagination = query.order_by(Attendance.timestamp.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        attendances = pagination.items
        
        return jsonify({
            'success': True,
            'data': [att.to_dict() for att in attendances],
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
        logger.error(f"Error getting attendance: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/attendance/<int:attendance_id>', methods=['GET'])
def get_attendance_record(attendance_id):
    """
    Get specific attendance record by ID.
    
    Args:
        attendance_id (int): Attendance record ID
        
    Returns:
        JSON: Attendance record details
    """
    try:
        attendance = Attendance.query.get(attendance_id)
        
        if not attendance:
            return jsonify({
                'success': False,
                'error': 'Attendance record not found'
            }), 404
        
        return jsonify({
            'success': True,
            'data': attendance.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance record {attendance_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/attendance/summary', methods=['GET'])
def get_attendance_summary():
    """
    Get attendance summary for a date range.
    
    Query parameters:
    - start_date: Start date (YYYY-MM-DD)
    - end_date: End date (YYYY-MM-DD)
    - employee_id: Filter by employee ID
    - department: Filter by department
    
    Returns:
        JSON: Attendance summary data
    """
    try:
        # Get query parameters
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        employee_id = request.args.get('employee_id', type=int)
        department = request.args.get('department', '')
        
        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'error': 'start_date and end_date are required'
            }), 400
        
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid date format. Use YYYY-MM-DD'
            }), 400
        
        # Build query
        query = db.session.query(
            Attendance.employee_id,
            db.func.count(db.func.distinct(db.func.date(Attendance.timestamp))).label('days_present'),
            db.func.count(Attendance.id).label('total_records'),
            db.func.avg(Attendance.face_similarity_score).label('avg_similarity')
        ).filter(
            db.func.date(Attendance.timestamp) >= start_dt,
            db.func.date(Attendance.timestamp) <= end_dt
        ).group_by(Attendance.employee_id)
        
        # Apply filters
        if employee_id:
            query = query.filter(Attendance.employee_id == employee_id)
        
        if department:
            query = query.join(Employee).filter(Employee.department == department)
        
        results = query.all()
        
        # Process results
        summary_data = []
        for result in results:
            employee = Employee.query.get(result.employee_id)
            if employee:
                summary_data.append({
                    'employee_id': result.employee_id,
                    'employee_name': employee.full_name,
                    'employee_number': employee.employee_number,
                    'department': employee.department,
                    'days_present': result.days_present,
                    'total_records': result.total_records,
                    'avg_similarity': float(result.avg_similarity) if result.avg_similarity else 0.0
                })
        
        return jsonify({
            'success': True,
            'data': {
                'start_date': start_date,
                'end_date': end_date,
                'summary': summary_data
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance summary: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

# Statistics API Endpoints

@api_bp.route('/stats/overview')
def get_overview_stats():
    """
    Get overview statistics.
    
    Returns:
        JSON: Overview statistics
    """
    try:
        # Get database statistics
        stats = get_database_stats(current_app)
        
        # Get today's date
        today = datetime.now().date()
        
        # Get today's attendance count
        today_attendance = Attendance.query.filter(
            db.func.date(Attendance.timestamp) == today
        ).count()
        
        # Get recent activity
        recent_attendances = Attendance.query.order_by(
            Attendance.timestamp.desc()
        ).limit(5).all()
        
        return jsonify({
            'success': True,
            'data': {
                'database_stats': stats,
                'today_attendance': today_attendance,
                'recent_activity': [att.to_dict() for att in recent_attendances],
                'current_time': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting overview stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/stats/employees')
def get_employee_stats():
    """
    Get employee statistics.
    
    Query parameters:
    - department: Filter by department
    
    Returns:
        JSON: Employee statistics
    """
    try:
        department = request.args.get('department', '')
        
        # Build query
        query = Employee.query.filter_by(is_active=True)
        
        if department:
            query = query.filter(Employee.department == department)
        
        # Get statistics
        total_employees = query.count()
        
        # Get employees with face embeddings
        employees_with_embeddings = db.session.query(
            db.func.count(db.func.distinct(FaceEmbedding.employee_id))
        ).scalar()
        
        # Get department breakdown
        dept_breakdown = db.session.query(
            Employee.department,
            db.func.count(Employee.id).label('count')
        ).filter_by(is_active=True).group_by(Employee.department).all()
        
        return jsonify({
            'success': True,
            'data': {
                'total_employees': total_employees,
                'employees_with_embeddings': employees_with_embeddings,
                'embedding_coverage': (employees_with_embeddings / total_employees * 100) if total_employees > 0 else 0,
                'department_breakdown': [
                    {'department': dept.department, 'count': dept.count}
                    for dept in dept_breakdown
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting employee stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@api_bp.route('/stats')
def get_stats():
    """Get system statistics."""
    try:
        total_employees = Employee.query.filter_by(is_active=True).count()
        total_attendance = Attendance.query.count()
        today_attendance = Attendance.query.filter(
            db.func.date(Attendance.timestamp) == datetime.now().date()
        ).count()
        
        return jsonify({
            'success': True,
            'data': {
                'total_employees': total_employees,
                'total_attendance': total_attendance,
                'today_attendance': today_attendance
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Error handlers

@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Resource not found'
    }), 404

@api_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@api_bp.route('/validate-face', methods=['POST'])
def validate_face():
    """
    Validate if an image contains a detectable face.
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image"
    }
    
    Returns:
        JSON: Face validation results
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: image'
            }), 400
        
        # Get image data
        image_data = data['image']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        # Convert base64 to image
        try:
            image = face_engine.base64_to_image(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid image data: {str(e)}'
            }), 400
        
        # Detect faces
        faces = face_engine.detect_faces(image)
        
        if not faces:
            return jsonify({
                'success': True,
                'faces_detected': 0,
                'faces': [],
                'message': 'No faces detected in image'
            })
        
        # Calculate best confidence
        best_confidence = max(face['confidence'] for face in faces)
        
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'best_confidence': best_confidence,
            'faces': [{
                'confidence': face['confidence'],
                'bbox': face['bbox']
            } for face in faces],
            'message': f'Found {len(faces)} face(s) with best confidence: {best_confidence:.3f}'
        })
        
    except Exception as e:
        logger.error(f"Error validating face: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500 
"""
Attendance Routes

This module contains routes for handling employee attendance including QR code scanning,
face verification, and time in/out recording.

Author: AI Assistant
Date: 2024
"""

from flask import Blueprint, render_template, request, jsonify, session, current_app
from datetime import datetime, timedelta
import cv2
import numpy as np
import base64
import logging

# Import models and utilities
from models.employee import Employee
from models.attendance import Attendance
from models.face_embedding import FaceEmbedding
from utils.face_recognition import face_engine
from utils.qr_code import qr_manager
from utils.database import db, commit_changes
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
attendance_bp = Blueprint('attendance', __name__)

@attendance_bp.route('/')
def attendance_home():
    """
    Main attendance page with time in/out buttons.
    
    Returns:
        str: Rendered attendance home page
    """
    return render_template('attendance/home.html')

@attendance_bp.route('/scan-qr')
def scan_qr():
    """
    QR code scanning page.
    
    Returns:
        str: Rendered QR scanning page
    """
    return render_template('attendance/scan_qr.html')

@attendance_bp.route('/verify-face')
def verify_face():
    """
    Face verification page.
    
    Returns:
        str: Rendered face verification page
    """
    return render_template('attendance/verify_face.html')

@attendance_bp.route('/api/process-qr-scan', methods=['POST'])
def process_qr_scan():
    """
    Process QR code scan and identify employee.
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image",
        "attendance_type": "in" or "out"
    }
    
    Returns:
        JSON: Processing results
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data or 'attendance_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: image and attendance_type'
            }), 400
        
        # Get data from request
        image_data = data['image']
        attendance_type = data['attendance_type'].lower()
        
        # Validate attendance type
        if attendance_type not in ['in', 'out']:
            return jsonify({
                'success': False,
                'error': 'Invalid attendance type. Must be "in" or "out"'
            }), 400
        
        # Convert base64 to image
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            image = face_engine.base64_to_image(image_data)
            logger.info(f"Successfully converted image: {image.shape}")
        except Exception as e:
            logger.error(f"Error converting base64 to image: {str(e)}")
            logger.error(f"Image data length: {len(image_data) if image_data else 0}")
            logger.error(f"Image data prefix: {image_data[:50] if image_data else 'None'}")
            return jsonify({
                'success': False,
                'error': f'Invalid image data: {str(e)}'
            }), 400
        
        # Process QR code scan
        logger.info("Processing QR code scan...")
        qr_result = qr_manager.process_qr_scan_for_attendance(image)
        logger.info(f"QR scan result: {qr_result}")
        
        if not qr_result['success']:
            logger.warning(f"QR scan failed: {qr_result['error']}")
            return jsonify({
                'success': False,
                'error': qr_result['error']
            }), 400
        
        # Get employee number from QR code
        employee_number = qr_result['employee_number']
        
        # Find employee in database
        employee = Employee.query.filter_by(employee_number=employee_number).first()
        
        if not employee:
            return jsonify({
                'success': False,
                'error': f'Employee with number {employee_number} not found'
            }), 404
        
        if not employee.is_active:
            return jsonify({
                'success': False,
                'error': 'Employee account is inactive'
            }), 400
        
        # Store employee info in session for face verification
        session['attendance_employee_id'] = employee.id
        session['attendance_type'] = attendance_type
        session['qr_scan_time'] = datetime.utcnow().isoformat()
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict(),
            'attendance_type': attendance_type,
            'message': f'QR code scanned successfully. Please show your face for verification.'
        })
        
    except Exception as e:
        logger.error(f"Error processing QR scan: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@attendance_bp.route('/api/process-face-verification', methods=['POST'])
def process_face_verification():
    """
    Process face verification for attendance.
    
    Expected JSON payload:
    {
        "images": ["base64_encoded_image1", "base64_encoded_image2", ...],
        "snapshot_count": 5
    }
    
    Returns:
        JSON: Verification results and attendance record
    """
    try:
        data = request.get_json()
        
        if not data or 'images' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: images'
            }), 400
        
        # Check if employee info is in session
        if 'attendance_employee_id' not in session:
            return jsonify({
                'success': False,
                'error': 'No employee identified. Please scan QR code first.'
            }), 400
        
        # Get employee info from session
        employee_id = session['attendance_employee_id']
        attendance_type = session['attendance_type']
        
        # Get employee
        employee = Employee.query.get(employee_id)
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Get stored face embeddings
        stored_embeddings = FaceEmbedding.get_employee_embeddings_as_arrays(employee_id)
        
        if not stored_embeddings:
            return jsonify({
                'success': False,
                'error': 'No face embeddings found for employee. Please register face first.'
            }), 400
        
        # Process each image
        images = data['images']
        snapshot_count = data.get('snapshot_count', Config.ATTENDANCE_SNAPSHOT_COUNT)
        
        if len(images) < snapshot_count:
            return jsonify({
                'success': False,
                'error': f'Insufficient images. Expected {snapshot_count}, got {len(images)}'
            }), 400
        
        # Process snapshots
        similarities = []
        detailed_results = []
        best_image = None
        best_similarity = 0.0
        
        for i, image_data in enumerate(images[:snapshot_count]):
            try:
                # Convert base64 to image
                image = face_engine.base64_to_image(image_data)
                
                # Process image for face recognition
                face_result = face_engine.process_image_for_attendance(image)
                
                if not face_result['success']:
                    logger.warning(f"Failed to process image {i}: {face_result['error']}")
                    detailed_results.append({
                        'image_index': i + 1,
                        'status': 'failed',
                        'error': face_result['error'],
                        'similarity': None,
                        'face_confidence': None
                    })
                    continue
                
                # Check face quality thresholds
                face_confidence = face_result['confidence']
                if face_confidence < Config.MIN_FACE_CONFIDENCE:
                    logger.warning(f"Image {i}: Face confidence too low: {face_confidence:.3f} < {Config.MIN_FACE_CONFIDENCE}")
                    detailed_results.append({
                        'image_index': i + 1,
                        'status': 'failed',
                        'error': f'Face confidence too low: {face_confidence:.1%} (required: {Config.MIN_FACE_CONFIDENCE:.1%})',
                        'similarity': None,
                        'face_confidence': face_confidence
                    })
                    continue
                
                # Compare with stored embeddings
                comparison = face_engine.compare_with_stored_embeddings(
                    face_result['embedding'], stored_embeddings
                )
                
                if comparison['success']:
                    avg_similarity = comparison['average_similarity']
                    similarities.append(avg_similarity)
                    
                    # Track best image
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_image = image_data
                    
                    # Store detailed results
                    detailed_results.append({
                        'image_index': i + 1,
                        'status': 'success',
                        'similarity': avg_similarity,
                        'face_confidence': face_result['confidence'],
                        'max_similarity': comparison['max_similarity'],
                        'min_similarity': comparison['min_similarity'],
                        'std_similarity': comparison['std_similarity']
                    })
                    
                    logger.info(f"Image {i+1}: Similarity {avg_similarity:.3f}, Face confidence {face_result['confidence']:.3f}")
                else:
                    detailed_results.append({
                        'image_index': i + 1,
                        'status': 'failed',
                        'error': comparison.get('error', 'Comparison failed'),
                        'similarity': None,
                        'face_confidence': face_result['confidence']
                    })
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                detailed_results.append({
                    'image_index': i + 1,
                    'status': 'error',
                    'error': str(e),
                    'similarity': None,
                    'face_confidence': None
                })
                continue
        
        # Check if we have enough valid similarities
        if len(similarities) < snapshot_count * 0.6:  # At least 60% success rate
            return jsonify({
                'success': False,
                'error': 'Insufficient valid face detections. Please try again.'
            }), 400
        
        # Calculate average similarity
        average_similarity = float(np.mean(similarities))  # Convert to Python float for JSON serialization
        logger.info(f"Face verification for employee {employee_id}: {len(similarities)} valid similarities, average: {average_similarity:.3f}")
        
        # Check if similarity meets threshold (use dynamic threshold from face engine)
        similarity_threshold = getattr(face_engine, 'similarity_threshold', Config.FACE_SIMILARITY_THRESHOLD)
        if average_similarity < similarity_threshold:
            logger.warning(f"Face verification failed for employee {employee_id}: {average_similarity:.3f} < {similarity_threshold}")
            logger.info(f"VERIFICATION FAILED - NO ATTENDANCE RECORD CREATED for employee {employee_id}")
            return jsonify({
                'success': False,
                'error': f'Face verification failed. Similarity score: {average_similarity:.3f} (required: {similarity_threshold:.3f})',
                'similarity_score': average_similarity,
                'detailed_results': detailed_results
            }), 400
        
        # Check for duplicate attendance (only if configured to do so)
        if not Config.ALLOW_DUPLICATE_ATTENDANCE and _check_duplicate_attendance(employee_id, attendance_type):
            return jsonify({
                'success': False,
                'error': f'Duplicate {attendance_type} detected within 5 minutes'
            }), 400
        
        # Create attendance record
        try:
            attendance = Attendance(
                employee_id=employee_id,
                type=attendance_type,
                face_similarity_score=average_similarity,
                face_snapshot_data=best_image,
                verification_method='face_qr',
                device_info=request.headers.get('User-Agent', 'Unknown'),
                notes=f'Processed {len(similarities)} snapshots, average similarity: {average_similarity:.3f}'
            )
            
            # Save to database
            db.session.add(attendance)
            commit_changes()
            logger.info(f"ATTENDANCE RECORD CREATED SUCCESSFULLY: ID {attendance.id}, Employee {employee_id}, Type {attendance_type}, Similarity {average_similarity:.3f}")
            
        except Exception as e:
            logger.error(f"Error creating attendance record: {str(e)}")
            db.session.rollback()
        
        # Check if we should add new face embedding for learning (use dynamic threshold)
        learning_threshold = getattr(face_engine, 'learning_threshold', Config.FACE_LEARNING_THRESHOLD)
        if (average_similarity > learning_threshold and 
            employee.can_add_face_embedding()):
            
            # Convert best image to embedding
            best_image_array = face_engine.base64_to_image(best_image)
            face_result = face_engine.process_image_for_attendance(best_image_array)
            
            if face_result['success']:
                # Create new face embedding
                face_embedding = FaceEmbedding(
                    employee_id=employee_id,
                    embedding=face_result['embedding'],
                    source_image_data=best_image,
                    similarity_score=average_similarity,
                    created_from_attendance=True,
                    attendance_id=attendance.id,
                    face_confidence=face_result['confidence']
                )
                
                db.session.add(face_embedding)
                commit_changes()
                
                logger.info(f"Added new face embedding for employee {employee_id}")
        
        # Clean up session
        session.pop('attendance_employee_id', None)
        session.pop('attendance_type', None)
        session.pop('qr_scan_time', None)
        
        return jsonify({
            'success': True,
            'attendance': attendance.to_dict(),
            'similarity_score': average_similarity,
            'detailed_results': detailed_results,
            'message': f'Time {attendance_type} recorded successfully for {employee.full_name}'
        })
        
    except Exception as e:
        logger.error(f"Error processing face verification: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@attendance_bp.route('/api/lookup-employee', methods=['POST'])
def lookup_employee():
    """
    Look up employee by employee number (for numeric keypad entry).
    
    Expected JSON payload:
    {
        "employee_number": "string",
        "attendance_type": "in" or "out"
    }
    
    Returns:
        JSON: Employee information if found
    """
    try:
        data = request.get_json()
        
        if not data or 'employee_number' not in data or 'attendance_type' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: employee_number and attendance_type'
            }), 400
        
        # Get data from request
        employee_number = data['employee_number'].strip()
        attendance_type = data['attendance_type'].lower()
        
        # Validate attendance type
        if attendance_type not in ['in', 'out']:
            return jsonify({
                'success': False,
                'error': 'Invalid attendance type. Must be "in" or "out"'
            }), 400
        
        # Validate employee number
        if not employee_number:
            return jsonify({
                'success': False,
                'error': 'Employee number cannot be empty'
            }), 400
        
        # Find employee in database
        employee = Employee.query.filter_by(employee_number=employee_number).first()
        
        if not employee:
            return jsonify({
                'success': False,
                'error': f'Employee with ID "{employee_number}" not found'
            }), 404
        
        if not employee.is_active:
            return jsonify({
                'success': False,
                'error': 'Employee account is inactive'
            }), 400
        
        # Store employee info in session for face verification
        session['attendance_employee_id'] = employee.id
        session['attendance_type'] = attendance_type
        session['qr_scan_time'] = datetime.utcnow().isoformat()
        
        return jsonify({
            'success': True,
            'employee': employee.to_dict(),
            'attendance_type': attendance_type,
            'message': f'Employee found: {employee.full_name}. Please show your face for verification.'
        })
        
    except Exception as e:
        logger.error(f"Error looking up employee: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@attendance_bp.route('/api/get-attendance-status')
def get_attendance_status():
    """
    Get current attendance status for display.
    
    Returns:
        JSON: Current attendance status
    """
    try:
        # Get today's attendance summary
        today = datetime.now().date()
        summary = Attendance.get_daily_summary(today)
        
        # Get active employees count
        active_employees = Employee.query.filter_by(is_active=True).count()
        
        # Get attendance records for today
        today_attendances = Attendance.query.filter(
            db.func.date(Attendance.timestamp) == today
        ).order_by(Attendance.timestamp.desc()).limit(10).all()
        
        return jsonify({
            'success': True,
            'today': today.isoformat(),
            'active_employees': active_employees,
            'attendance_summary': [
                {
                    'employee_id': s.employee_id,
                    'first_in': s.first_in.isoformat() if s.first_in else None,
                    'last_out': s.last_out.isoformat() if s.last_out else None,
                    'total_records': s.total_records
                }
                for s in summary
            ],
            'recent_attendances': [att.to_dict() for att in today_attendances]
        })
        
    except Exception as e:
        logger.error(f"Error getting attendance status: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

def _check_duplicate_attendance(employee_id: int, attendance_type: str) -> bool:
    """
    Check for duplicate attendance within 5 minutes.
    
    Args:
        employee_id (int): Employee ID
        attendance_type (str): Type of attendance ('in' or 'out')
        
    Returns:
        bool: True if duplicate found, False otherwise
    """
    try:
        # Check for attendance records in the last 5 minutes
        five_minutes_ago = datetime.utcnow() - timedelta(minutes=5)
        
        recent_attendance = Attendance.query.filter(
            Attendance.employee_id == employee_id,
            Attendance.type == attendance_type,
            Attendance.timestamp >= five_minutes_ago
        ).first()
        
        return recent_attendance is not None
        
    except Exception as e:
        logger.error(f"Error checking duplicate attendance: {str(e)}")
        return False 
"""
Admin Routes

This module contains routes for administrative functions including DTR generation,
system management, and administrative reports.

Author: AI Assistant
Date: 2024
"""

from flask import Blueprint, render_template, request, jsonify, send_file
from datetime import datetime, timedelta
import io
import logging
import os

# Import models and utilities
from models.employee import Employee
from models.attendance import Attendance
from models.face_embedding import FaceEmbedding
from utils.database import db, get_database_stats
from utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/dashboard')
def admin_dashboard():
    """
    Admin dashboard page.
    
    Returns:
        str: Rendered admin dashboard
    """
    return render_template('admin/dashboard.html')

@admin_bp.route('/dtr-generation', methods=['GET', 'POST'])
def dtr_generation():
    """
    DTR generation page. Handles both displaying the form and processing it.
    """
    if request.method == 'POST':
        employee_id = request.form.get('employee_id')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')

        # Validate dates
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            # Handle error appropriately
            employees = Employee.query.order_by(Employee.last_name).all()
            departments = sorted(list(set(emp.department for emp in employees if emp.department)))
            return render_template('admin/dtr.html', 
                                   employees=employees, 
                                   departments=departments,
                                   error="Invalid date format.")

        # Fetch employee
        employee = Employee.query.get(employee_id)
        
        # Fetch attendance records
        attendances = Attendance.query.filter(
            Attendance.employee_id == employee_id,
            db.func.date(Attendance.timestamp) >= start_date,
            db.func.date(Attendance.timestamp) <= end_date
        ).order_by(Attendance.timestamp).all()

        # Process attendances
        daily_attendance = {}
        for att in attendances:
            day = att.timestamp.date()
            if day not in daily_attendance:
                daily_attendance[day] = {'in': None, 'out': None, 'in_image': None, 'out_image': None, 'all_logs': []}

            # Store all logs for the detailed view
            daily_attendance[day]['all_logs'].append({
                'timestamp': att.timestamp,
                'status': att.type,
                'image_path': att.face_snapshot_data
            })

            if att.type.upper() == 'IN':
                if daily_attendance[day]['in'] is None or att.timestamp < daily_attendance[day]['in']:
                    daily_attendance[day]['in'] = att.timestamp
                    daily_attendance[day]['in_image'] = att.face_snapshot_data
            elif att.type.upper() == 'OUT':
                if daily_attendance[day]['out'] is None or att.timestamp > daily_attendance[day]['out']:
                    daily_attendance[day]['out'] = att.timestamp
                    daily_attendance[day]['out_image'] = att.face_snapshot_data
        
        # Calculate total hours
        summary_data = []
        for day, data in sorted(daily_attendance.items()):
            total_hours = 0
            if data['in'] and data['out']:
                duration = data['out'] - data['in']
                total_hours = round(duration.total_seconds() / 3600, 2)
            
            summary_data.append({
                'date': day,
                'time_in': data['in'].strftime('%H:%M:%S') if data['in'] else 'N/A',
                'time_out': data['out'].strftime('%H:%M:%S') if data['out'] else 'N/A',
                'total_hours': total_hours
            })

        employees = Employee.query.order_by(Employee.last_name).all()
        departments = sorted(list(set(emp.department for emp in employees if emp.department)))
        
        return render_template('admin/dtr.html', 
                               employees=employees,
                               departments=departments,
                               selected_employee=employee,
                               summary_data=summary_data,
                               detailed_data=daily_attendance,
                               start_date=start_date.strftime('%Y-%m-%d'),
                               end_date=end_date.strftime('%Y-%m-%d'))


    # For GET request
    employees = Employee.query.order_by(Employee.last_name).all()
    departments = sorted(list(set(emp.department for emp in employees if emp.department)))
    return render_template('admin/dtr.html', employees=employees, departments=departments)

@admin_bp.route('/reports')
def reports():
    """
    Reports page.
    
    Returns:
        str: Rendered reports page
    """
    return render_template('admin/reports.html')

@admin_bp.route('/system-settings')
def system_settings():
    """
    System settings page.
    
    Returns:
        str: Rendered system settings page
    """
    return render_template('admin/system_settings.html')

@admin_bp.route('/api/dashboard-stats')
def api_dashboard_stats():
    """
    Get dashboard statistics.
    
    Returns:
        JSON: Dashboard statistics
    """
    try:
        # Get database statistics
        stats = get_database_stats(current_app)
        
        # Get today's date
        today = datetime.now().date()
        
        # Get recent attendance records
        recent_attendances = Attendance.query.filter(
            db.func.date(Attendance.timestamp) == today
        ).order_by(Attendance.timestamp.desc()).limit(10).all()
        
        # Get department statistics
        departments = db.session.query(
            Employee.department,
            db.func.count(Employee.id).label('count')
        ).filter_by(is_active=True).group_by(Employee.department).all()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_attendances': [att.to_dict() for att in recent_attendances],
            'departments': [
                {'department': dept.department, 'count': dept.count}
                for dept in departments
            ],
            'today': today.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@admin_bp.route('/api/generate-dtr', methods=['POST'])
def api_generate_dtr():
    """
    Generate Daily Time Record (DTR) PDF.
    
    Expected JSON payload:
    {
        "employee_id": 1,
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "include_snapshots": true
    }
    
    Returns:
        JSON: DTR generation results or PDF file
    """
    try:
        data = request.get_json()
        
        if not data or 'employee_id' not in data or 'start_date' not in data or 'end_date' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: employee_id, start_date, end_date'
            }), 400
        
        # Get employee
        employee = Employee.query.get(data['employee_id'])
        if not employee:
            return jsonify({
                'success': False,
                'error': 'Employee not found'
            }), 404
        
        # Parse dates
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid date format. Use YYYY-MM-DD'
            }), 400
        
        # Get attendance records
        attendances = Attendance.get_employee_attendance_by_date_range(
            employee.id, start_date, end_date
        )
        
        if not attendances:
            return jsonify({
                'success': False,
                'error': 'No attendance records found for the specified date range'
            }), 404
        
        # Generate DTR PDF
        include_snapshots = data.get('include_snapshots', True)
        pdf_data = _generate_dtr_pdf(employee, attendances, start_date, end_date, include_snapshots)
        
        # Return PDF file
        return send_file(
            io.BytesIO(pdf_data),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'DTR_{employee.employee_number}_{start_date}_{end_date}.pdf'
        )
        
    except Exception as e:
        logger.error(f"Error generating DTR: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@admin_bp.route('/api/attendance-report', methods=['POST'])
def api_attendance_report():
    """
    Generate attendance report.
    
    Expected JSON payload:
    {
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "department": "IT",
        "report_type": "daily" or "summary"
    }
    
    Returns:
        JSON: Attendance report data
    """
    try:
        data = request.get_json()
        
        if not data or 'start_date' not in data or 'end_date' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required data: start_date, end_date'
            }), 400
        
        # Parse dates
        try:
            start_date = datetime.strptime(data['start_date'], '%Y-%m-%d').date()
            end_date = datetime.strptime(data['end_date'], '%Y-%m-%d').date()
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid date format. Use YYYY-MM-DD'
            }), 400
        
        # Get report type
        report_type = data.get('report_type', 'summary')
        department = data.get('department', '')
        
        # Generate report
        if report_type == 'daily':
            report_data = _generate_daily_report(start_date, end_date, department)
        else:
            report_data = _generate_summary_report(start_date, end_date, department)
        
        return jsonify({
            'success': True,
            'report': report_data,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'report_type': report_type
        })
        
    except Exception as e:
        logger.error(f"Error generating attendance report: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@admin_bp.route('/api/employee-stats')
def api_employee_stats():
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
        
        # Get recent registrations
        recent_registrations = Employee.query.filter_by(is_active=True).order_by(
            Employee.created_at.desc()
        ).limit(5).all()
        
        return jsonify({
            'success': True,
            'total_employees': total_employees,
            'employees_with_embeddings': employees_with_embeddings,
            'embedding_coverage': (employees_with_embeddings / total_employees * 100) if total_employees > 0 else 0,
            'department_breakdown': [
                {'department': dept.department, 'count': dept.count}
                for dept in dept_breakdown
            ],
            'recent_registrations': [emp.to_dict() for emp in recent_registrations]
        })
        
    except Exception as e:
        logger.error(f"Error getting employee stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@admin_bp.route('/api/system-info')
def api_system_info():
    """
    Get system information and configuration.
    
    Returns:
        JSON: System information
    """
    try:
        # Get database statistics
        stats = get_database_stats(current_app)
        
        # Get configuration info
        config_info = {
            'face_similarity_threshold': Config.FACE_SIMILARITY_THRESHOLD,
            'face_learning_threshold': Config.FACE_LEARNING_THRESHOLD,
            'max_face_embeddings': Config.MAX_FACE_EMBEDDINGS_PER_EMPLOYEE,
            'attendance_snapshot_count': Config.ATTENDANCE_SNAPSHOT_COUNT,
            'registration_snapshot_count': Config.REGISTRATION_SNAPSHOT_COUNT,
            'upload_folder': Config.UPLOAD_FOLDER,
            'allowed_extensions': list(Config.ALLOWED_EXTENSIONS)
        }
        
        # Get system status
        system_status = {
            'database_connected': True,  # If we get here, DB is connected
            'upload_folder_exists': os.path.exists(Config.UPLOAD_FOLDER),
            'logs_folder_exists': os.path.exists(os.path.dirname(Config.LOG_FILE))
        }
        
        return jsonify({
            'success': True,
            'stats': stats,
            'config': config_info,
            'system_status': system_status,
            'current_time': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

def _generate_dtr_pdf(employee, attendances, start_date, end_date, include_snapshots=True):
    """
    Generate DTR PDF for an employee.
    
    Args:
        employee: Employee object
        attendances: List of attendance records
        start_date: Start date
        end_date: End date
        include_snapshots: Whether to include face snapshots
        
    Returns:
        bytes: PDF data
    """
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import base64
    from PIL import Image
    import io
    
    # Create PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    # Build story
    story = []
    
    # Title
    title = Paragraph(f"Daily Time Record<br/>{employee.full_name}", title_style)
    story.append(title)
    
    # Employee information
    emp_info = [
        ['Employee Number:', employee.employee_number],
        ['Name:', employee.full_name],
        ['Department:', employee.department],
        ['Position:', employee.position],
        ['Period:', f"{start_date} to {end_date}"]
    ]
    
    emp_table = Table(emp_info, colWidths=[2*inch, 4*inch])
    emp_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(emp_table)
    story.append(Spacer(1, 20))
    
    # Attendance table
    attendance_data = [['Date', 'Time In', 'Time Out', 'Total Hours', 'Status']]
    
    current_date = start_date
    while current_date <= end_date:
        # Get attendances for this date
        day_attendances = [att for att in attendances if att.date == current_date]
        
        if day_attendances:
            # Find time in and out
            time_in = None
            time_out = None
            
            for att in day_attendances:
                if att.is_time_in:
                    time_in = att.time
                elif att.is_time_out:
                    time_out = att.time
            
            # Calculate total hours
            total_hours = "N/A"
            if time_in and time_out:
                # Calculate hours difference
                time_diff = datetime.combine(current_date, time_out) - datetime.combine(current_date, time_in)
                total_hours = f"{time_diff.total_seconds() / 3600:.2f}"
            
            status = "Complete" if time_in and time_out else "Incomplete"
            
            attendance_data.append([
                current_date.strftime('%Y-%m-%d'),
                time_in.strftime('%H:%M:%S') if time_in else "N/A",
                time_out.strftime('%H:%M:%S') if time_out else "N/A",
                total_hours,
                status
            ])
        else:
            attendance_data.append([
                current_date.strftime('%Y-%m-%d'),
                "N/A", "N/A", "N/A", "No Record"
            ])
        
        current_date += timedelta(days=1)
    
    # Create attendance table
    att_table = Table(attendance_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
    att_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(att_table)
    
    # Build PDF
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def _generate_daily_report(start_date, end_date, department=''):
    """
    Generate daily attendance report.
    
    Args:
        start_date: Start date
        end_date: End date
        department: Department filter
        
    Returns:
        dict: Daily report data
    """
    # Build query
    query = db.session.query(
        Attendance.employee_id,
        db.func.date(Attendance.timestamp).label('date'),
        db.func.min(Attendance.timestamp).label('first_in'),
        db.func.max(Attendance.timestamp).label('last_out'),
        db.func.count(Attendance.id).label('total_records')
    ).group_by(
        Attendance.employee_id,
        db.func.date(Attendance.timestamp)
    ).filter(
        db.func.date(Attendance.timestamp) >= start_date,
        db.func.date(Attendance.timestamp) <= end_date
    )
    
    # Apply department filter
    if department:
        query = query.join(Employee).filter(Employee.department == department)
    
    results = query.all()
    
    # Process results
    daily_data = {}
    for result in results:
        date_str = result.date.isoformat()
        if date_str not in daily_data:
            daily_data[date_str] = []
        
        # Get employee info
        employee = Employee.query.get(result.employee_id)
        if employee:
            daily_data[date_str].append({
                'employee_id': result.employee_id,
                'employee_name': employee.full_name,
                'employee_number': employee.employee_number,
                'department': employee.department,
                'first_in': result.first_in.isoformat() if result.first_in else None,
                'last_out': result.last_out.isoformat() if result.last_out else None,
                'total_records': result.total_records
            })
    
    return {
        'type': 'daily',
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'department': department,
        'daily_data': daily_data
    }

def _generate_summary_report(start_date, end_date, department=''):
    """
    Generate summary attendance report.
    
    Args:
        start_date: Start date
        end_date: End date
        department: Department filter
        
    Returns:
        dict: Summary report data
    """
    # Build query
    query = db.session.query(
        Attendance.employee_id,
        db.func.count(db.func.distinct(db.func.date(Attendance.timestamp))).label('days_present'),
        db.func.count(Attendance.id).label('total_records'),
        db.func.avg(Attendance.face_similarity_score).label('avg_similarity')
    ).filter(
        db.func.date(Attendance.timestamp) >= start_date,
        db.func.date(Attendance.timestamp) <= end_date
    ).group_by(Attendance.employee_id)
    
    # Apply department filter
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
    
    return {
        'type': 'summary',
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'department': department,
        'summary_data': summary_data
    } 
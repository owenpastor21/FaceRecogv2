"""
Database Initialization Utility

This module handles database initialization, table creation, and database
connection management for the facial recognition attendance system.

Author: AI Assistant
Date: 2024
"""

# Import shared database instance and models
from models import db, Employee, Attendance, FaceEmbedding

def init_db(app):
    """
    Initialize the database with the Flask app.
    
    This function sets up the database connection, creates all tables,
    and ensures the database is ready for use.
    
    Args:
        app: Flask application instance
    """
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Log database initialization
        app.logger.info("Database initialized successfully")
        
        # Check if we need to create any initial data
        _create_initial_data(app)

def _create_initial_data(app):
    """
    Create initial data if the database is empty.
    
    Args:
        app: Flask application instance
    """
    # Check if we have any employees
    if Employee.query.count() == 0:
        app.logger.info("No employees found. Database is ready for first use.")
    
    # Check if we have any attendance records
    if Attendance.query.count() == 0:
        app.logger.info("No attendance records found. Database is ready for first use.")
    
    # Check if we have any face embeddings
    if FaceEmbedding.query.count() == 0:
        app.logger.info("No face embeddings found. Database is ready for first use.")

def get_db_session():
    """
    Get a database session for manual database operations.
    
    Returns:
        SQLAlchemy session object
    """
    return db.session

def commit_changes():
    """
    Commit pending database changes.
    
    This function commits all pending changes in the current session.
    """
    try:
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        raise e

def rollback_changes():
    """
    Rollback pending database changes.
    
    This function rolls back all pending changes in the current session.
    """
    db.session.rollback()

def close_session():
    """
    Close the current database session.
    
    This function closes the current session and releases resources.
    """
    db.session.close()

def reset_database(app):
    """
    Reset the database by dropping all tables and recreating them.
    
    WARNING: This will delete all data in the database!
    
    Args:
        app: Flask application instance
    """
    with app.app_context():
        # Drop all tables
        db.drop_all()
        
        # Recreate all tables
        db.create_all()
        
        app.logger.warning("Database has been reset - all data has been deleted!")

def backup_database(app, backup_path):
    """
    Create a backup of the database.
    
    Args:
        app: Flask application instance
        backup_path (str): Path where to save the backup
        
    Returns:
        bool: True if backup was successful, False otherwise
    """
    import shutil
    import os
    from datetime import datetime
    
    try:
        # Get the database file path
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        
        if os.path.exists(db_path):
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"attendance_backup_{timestamp}.db"
            backup_file_path = os.path.join(backup_path, backup_filename)
            
            # Ensure backup directory exists
            os.makedirs(backup_path, exist_ok=True)
            
            # Copy database file
            shutil.copy2(db_path, backup_file_path)
            
            app.logger.info(f"Database backup created: {backup_file_path}")
            return True
        else:
            app.logger.error(f"Database file not found: {db_path}")
            return False
            
    except Exception as e:
        app.logger.error(f"Failed to create database backup: {str(e)}")
        return False

def get_database_stats(app):
    """
    Get database statistics.
    
    Args:
        app: Flask application instance
        
    Returns:
        dict: Dictionary containing database statistics
    """
    with app.app_context():
        stats = {
            'total_employees': Employee.query.count(),
            'active_employees': Employee.query.filter_by(is_active=True).count(),
            'total_attendance_records': Attendance.query.count(),
            'total_face_embeddings': FaceEmbedding.query.count(),
            'attendance_records_today': Attendance.query.filter(
                db.func.date(Attendance.timestamp) == db.func.date(db.func.now())
            ).count(),
            'employees_with_embeddings': db.session.query(
                db.func.count(db.func.distinct(FaceEmbedding.employee_id))
            ).scalar()
        }
        
        return stats 
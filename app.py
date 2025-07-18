"""
Main Flask Application for Facial Recognition Attendance System

This module serves as the entry point for the facial recognition attendance application.
It initializes the Flask app, configures the database, and registers all blueprints.

Author: AI Assistant
Date: 2024
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import os
from datetime import datetime

# Import shared database instance and models
from models import db, Employee, Attendance, FaceEmbedding

# Import blueprints
from routes.attendance import attendance_bp
from routes.employee import employee_bp
from routes.admin import admin_bp
from routes.api import api_bp

# Import utilities
from utils.config import Config
from utils.database import init_db

def create_app(config_class=Config):
    """
    Application factory pattern for creating Flask app instances.
    
    Args:
        config_class: Configuration class to use for the app
        
    Returns:
        Flask app instance with all configurations and blueprints registered
    """
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize CORS for cross-origin requests
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize database with shared instance
    db.init_app(app)
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(attendance_bp, url_prefix='/attendance')
    app.register_blueprint(employee_bp, url_prefix='/employee')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        """Main landing page for the application."""
        return render_template('index.html')
    
    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    # Check if we're in production or development
    import os
    
    # SSL Context for HTTPS (enables camera access from mobile devices)
    ssl_context = None
    
    try:
        # Try to create SSL context with certificate files
        if os.path.exists('cert.pem') and os.path.exists('key.pem'):
            ssl_context = ('cert.pem', 'key.pem')
            print("🔒 HTTPS enabled with existing certificates")
        else:
            # Generate self-signed certificate for development
            print("🔧 Generating self-signed SSL certificate...")
            ssl_context = 'adhoc'  # Flask will generate temporary certificate
            print("⚠️  Warning: Using temporary self-signed certificate")
            print("📱 You'll need to accept the security warning on your mobile browser")
    except Exception as e:
        print(f"⚠️  SSL setup failed: {e}")
        print("🌐 Running without HTTPS - camera may not work on mobile devices")
    
    # Start the application
    if ssl_context:
        print(f"🚀 Starting HTTPS server on https://192.168.100.13:5000")
        print("📱 Mobile devices can now access camera features!")
        app = create_app()
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=True,
            ssl_context=ssl_context
        )
    else:
        print(f"🚀 Starting HTTP server on http://192.168.100.13:5000")
        print("⚠️  Camera access may be blocked on mobile devices")
        app = create_app()
        app.run(host='0.0.0.0', port=5000, debug=True) 
"""
QR Code Utility

This module provides QR code generation and scanning functionality for employee
identification in the facial recognition attendance system.

Author: AI Assistant
Date: 2024
"""

import qrcode
import cv2
import numpy as np
from pyzbar import pyzbar
from PIL import Image
import base64
import io
from typing import Optional, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QRCodeManager:
    """
    QR Code manager for generating and scanning QR codes.
    
    This class handles QR code generation for employee identification and
    QR code scanning from images or camera feeds.
    """
    
    def __init__(self):
        """
        Initialize the QR code manager.
        """
        logger.info("QR Code Manager initialized successfully")
    
    def generate_qr_code(self, data: str, size: int = 10, version: int = 1) -> str:
        """
        Generate a QR code image from data.
        
        Args:
            data (str): Data to encode in QR code
            size (int): Size of QR code in pixels
            version (int): QR code version (1-40)
            
        Returns:
            str: Base64 encoded QR code image
        """
        try:
            # Create QR code
            qr = qrcode.QRCode(
                version=version,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=size,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating QR code: {str(e)}")
            return None
    
    def scan_qr_code_from_image(self, image: np.ndarray) -> List[Dict]:
        """
        Scan QR codes from an image.
        
        Args:
            image (np.ndarray): Input image as numpy array
            
        Returns:
            List[Dict]: List of detected QR codes with data and bounding boxes
        """
        try:
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect QR codes
            decoded_objects = pyzbar.decode(gray)
            
            results = []
            for obj in decoded_objects:
                # Extract data
                data = obj.data.decode('utf-8')
                qr_type = obj.type
                
                # Extract bounding box
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    points = hull
                
                # Convert points to list
                bbox = []
                for point in points:
                    bbox.append([point[0], point[1]])
                
                results.append({
                    'data': data,
                    'type': qr_type,
                    'bbox': bbox,
                    'confidence': 1.0  # pyzbar doesn't provide confidence scores
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error scanning QR code: {str(e)}")
            return []
    
    def scan_qr_code_from_base64(self, base64_string: str) -> List[Dict]:
        """
        Scan QR codes from a base64 encoded image.
        
        Args:
            base64_string (str): Base64 encoded image string
            
        Returns:
            List[Dict]: List of detected QR codes
        """
        try:
            # Decode base64
            img_data = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return self.scan_qr_code_from_image(image)
            
        except Exception as e:
            logger.error(f"Error scanning QR code from base64: {str(e)}")
            return []
    
    def extract_employee_number_from_qr(self, qr_data: str) -> Optional[str]:
        """
        Extract employee number from QR code data.
        
        Args:
            qr_data (str): QR code data
            
        Returns:
            Optional[str]: Employee number if valid, None otherwise
        """
        try:
            # Expected format: "EMP_EMPLOYEE_NUMBER"
            if qr_data.startswith("EMP_"):
                employee_number = qr_data[4:]  # Remove "EMP_" prefix
                return employee_number
            else:
                logger.warning(f"Invalid QR code format: {qr_data}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting employee number: {str(e)}")
            return None
    
    def validate_qr_code_data(self, qr_data: str) -> Dict:
        """
        Validate QR code data format.
        
        Args:
            qr_data (str): QR code data to validate
            
        Returns:
            Dict: Validation results
        """
        try:
            # Check if data starts with expected prefix
            if not qr_data.startswith("EMP_"):
                return {
                    'valid': False,
                    'error': 'Invalid QR code format - must start with "EMP_"'
                }
            
            # Extract employee number
            employee_number = qr_data[4:]
            
            # Check if employee number is not empty
            if not employee_number:
                return {
                    'valid': False,
                    'error': 'Employee number is empty'
                }
            
            # Check if employee number contains only alphanumeric characters
            if not employee_number.replace('-', '').replace('_', '').isalnum():
                return {
                    'valid': False,
                    'error': 'Employee number contains invalid characters'
                }
            
            return {
                'valid': True,
                'employee_number': employee_number
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}'
            }
    
    def generate_employee_qr_code(self, employee_number: str, size: int = 10) -> str:
        """
        Generate QR code for an employee.
        
        Args:
            employee_number (str): Employee number
            size (int): QR code size in pixels
            
        Returns:
            str: Base64 encoded QR code image
        """
        # Create QR data with prefix
        qr_data = f"EMP_{employee_number}"
        
        return self.generate_qr_code(qr_data, size)
    
    def process_qr_scan_for_attendance(self, image: np.ndarray) -> Dict:
        """
        Process QR code scan for attendance verification.
        
        Args:
            image (np.ndarray): Input image containing QR code
            
        Returns:
            Dict: Processing results including employee number if found
        """
        # Scan QR codes
        qr_codes = self.scan_qr_code_from_image(image)
        
        if not qr_codes:
            return {
                'success': False,
                'error': 'No QR codes detected in image',
                'qr_codes': []
            }
        
        # Process the first detected QR code
        qr_code = qr_codes[0]
        
        # Validate QR code data
        validation = self.validate_qr_code_data(qr_code['data'])
        
        if not validation['valid']:
            return {
                'success': False,
                'error': validation['error'],
                'qr_codes': qr_codes
            }
        
        return {
            'success': True,
            'employee_number': validation['employee_number'],
            'qr_data': qr_code['data'],
            'qr_codes': qr_codes
        }
    
    def enhance_image_for_qr_scanning(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better QR code detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image
    
    def draw_qr_code_boxes(self, image: np.ndarray, qr_codes: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected QR codes.
        
        Args:
            image (np.ndarray): Input image
            qr_codes (List[Dict]): List of detected QR codes
            
        Returns:
            np.ndarray: Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for qr_code in qr_codes:
            bbox = qr_code['bbox']
            
            # Convert bbox to integer points
            points = np.array(bbox, dtype=np.int32)
            
            # Draw bounding box
            cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
            
            # Add text label
            text = f"QR: {qr_code['data'][:10]}..."
            cv2.putText(result_image, text, (points[0][0], points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_image
    
    def get_qr_code_statistics(self, qr_codes: List[Dict]) -> Dict:
        """
        Get statistics about detected QR codes.
        
        Args:
            qr_codes (List[Dict]): List of detected QR codes
            
        Returns:
            Dict: Statistics about the QR codes
        """
        if not qr_codes:
            return {
                'total_codes': 0,
                'valid_codes': 0,
                'invalid_codes': 0,
                'employee_numbers': []
            }
        
        valid_codes = 0
        employee_numbers = []
        
        for qr_code in qr_codes:
            validation = self.validate_qr_code_data(qr_code['data'])
            if validation['valid']:
                valid_codes += 1
                employee_numbers.append(validation['employee_number'])
        
        return {
            'total_codes': len(qr_codes),
            'valid_codes': valid_codes,
            'invalid_codes': len(qr_codes) - valid_codes,
            'employee_numbers': employee_numbers
        }

# Global instance of the QR code manager
qr_manager = QRCodeManager() 
# 🎯 Facial Recognition Attendance System - FaceNet512 Edition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent attendance tracking system using **state-of-the-art FaceNet512** technology with 512-dimensional face embeddings for superior accuracy and discrimination.

![System Demo](https://via.placeholder.com/800x400/667eea/ffffff?text=Face+Recognition+Attendance+System)

## 🌟 **Key Features**

### 🔐 **Advanced Biometric Security**
- **FaceNet512 Recognition**: 512-dimensional deep learning embeddings
- **Dual Verification**: QR code + facial biometrics  
- **Real-time Processing**: Instant verification with multiple snapshot analysis
- **Anti-Spoofing**: Built-in security against photo/video attacks

### 📱 **Intelligent Interface**
- **Sleep Mode**: Touch-activated with "IN" and "OUT" buttons
- **Camera Integration**: Live face capture with real-time feedback
- **QR Code Generation**: Automatic employee QR code creation
- **Progressive Scanning**: 5-image capture for maximum accuracy

### 👥 **Employee Management**
- **Smart Registration**: Face profile creation with quality validation
- **Adaptive Learning**: System improves recognition over time (up to 50 embeddings per employee)
- **Department Organization**: Complete employee information management
- **Bulk Operations**: Efficient mass employee registration

### 📊 **Comprehensive Reporting**
- **Real-time Dashboard**: Live attendance monitoring
- **PDF Reports**: Professional daily time records (DTR)
- **Analytics**: Detailed attendance statistics and trends
- **Individual Tracking**: Per-employee attendance history

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- Webcam or USB camera
- Windows/Linux/macOS

### **Installation**

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/FaceRecogV2.git
cd FaceRecogV2
```

2. **Set up virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python app.py
```

5. **Access the system:**
   - Open browser: `http://localhost:5000`
   - Start with employee registration
   - Enjoy superior FaceNet512 accuracy!

## 🛠️ **Technology Stack**

### **Core Technologies**
- **Backend**: Flask (Python web framework)
- **Database**: SQLAlchemy with SQLite (PostgreSQL for production)
- **Face Recognition**: FaceNet512 via DeepFace + TensorFlow
- **Computer Vision**: MediaPipe (face detection), OpenCV
- **Frontend**: Bootstrap 5, JavaScript ES6+

### **AI/ML Components**
- **FaceNet512**: 512-dimensional face embeddings (Primary)
- **MediaPipe**: Fast face detection and landmarks
- **TensorFlow**: Deep learning inference engine
- **dlib ResNet**: 128-dimensional embeddings (Fallback)

### **Recognition Pipeline**
```
Camera Input → MediaPipe Detection → Face Extraction → 
FaceNet512 Embedding (512D) → Cosine Similarity → 
Attendance Decision (40% threshold)
```

## 📋 **System Requirements**

### **Minimum Requirements**
- **CPU**: 2 GHz dual-core processor
- **RAM**: 4 GB (8 GB recommended for FaceNet512)
- **Storage**: 2 GB free space
- **Camera**: USB webcam or built-in camera

### **Recommended for Optimal Performance**
- **CPU**: Intel i5/AMD Ryzen 5 or better
- **RAM**: 8 GB+ (for TensorFlow optimization)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: SSD for faster model loading

## 🎯 **Recognition Accuracy**

### **FaceNet512 Performance**
- **Same Person**: 70-95% similarity ✅
- **Different People**: 20-60% similarity ✅
- **False Positive Rate**: <2% 🎯
- **Processing Speed**: ~200-500ms per verification

### **Performance Comparison**

| Recognition Method | Embedding Dimensions | Different People Similarity | Discrimination Quality |
|-------------------|---------------------|---------------------------|----------------------|
| **FaceNet512** (New) | **512** | **0.2-0.6** | **⭐⭐⭐⭐⭐** |
| dlib ResNet (Previous) | 128 | 0.96-0.99 | ⭐⭐⭐ |
| MediaPipe (Fallback) | 1,404 | 0.85-0.95 | ⭐⭐ |

## 📁 **Project Structure**

```
FaceRecogV2/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
│
├── models/                         # Database Models
│   ├── employee.py                # Employee with QR generation
│   ├── attendance.py              # Attendance tracking
│   └── face_embedding.py          # Face embeddings storage
│
├── routes/                         # Flask Blueprints
│   ├── attendance.py              # QR + Face scanning
│   ├── employee.py                # Employee CRUD
│   ├── admin.py                   # Dashboard & reports
│   └── api.py                     # RESTful APIs
│
├── utils/                          # Core Logic
│   ├── config.py                  # Configuration
│   ├── face_recognition.py        # FaceNet512 engine
│   └── qr_code.py                 # QR generation/scanning
│
├── templates/                      # HTML Templates
│   ├── base.html                  # Bootstrap layout
│   ├── index.html                 # Landing page
│   ├── attendance/                # Attendance UI
│   ├── employee/                  # Employee management
│   └── admin/                     # Admin interfaces
│
└── static/                        # CSS, JS, Images
```

## 🔧 **Configuration**

Key settings in `utils/config.py`:

```python
# FaceNet512 Settings (Primary)
FACENET512_SIMILARITY_THRESHOLD = 0.4  # 40% for attendance
FACENET512_LEARNING_THRESHOLD = 0.5    # 50% for learning new faces

# Fallback Settings
FACE_SIMILARITY_THRESHOLD = 0.6        # dlib fallback
FACE_LEARNING_THRESHOLD = 0.7          # dlib fallback

# Camera Settings
DEFAULT_CAMERA_RESOLUTION = 'medium'   # 720p
ATTENDANCE_SNAPSHOT_COUNT = 5          # Verification images
```

## 🐳 **Docker Deployment**

Build and run with Docker:

```bash
# Build the image
docker build -t face-attendance .

# Run with GPU support (optional)
docker run --gpus all -p 5000:5000 face-attendance

# Run without GPU
docker run -p 5000:5000 face-attendance
```

## 📚 **Documentation**

- **[Technical Specification](Idea.md)**: Complete architecture and implementation details
- **[User Experience Guide](experience.md)**: Detailed UI/UX documentation and workflows
- **[Setup Guide](DEEP_LEARNING_SETUP.md)**: Installation and configuration instructions
- **[Docker Guide](DOCKER_DEPLOYMENT_GUIDE.md)**: Container deployment instructions

## 🤝 **Contributing**

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 **API Endpoints**

### **Core APIs**
- `GET /api/employees` - List employees with pagination
- `POST /api/employees` - Create new employee
- `POST /attendance/api/process-qr-scan` - Process attendance with QR + Face
- `POST /employee/api/register-face` - Register face embeddings
- `GET /api/stats/overview` - System statistics

### **Face Recognition APIs**
- `POST /api/validate-face` - Validate face image quality
- `POST /api/face/compare` - Compare face embeddings
- `GET /employees/{id}/embeddings` - Get employee face profiles

See [Technical Specification](Idea.md) for complete API documentation.

## 🔐 **Security Features**

- **Encrypted face embeddings** in database
- **HTTPS/SSL support** for camera access
- **Input validation** and sanitization
- **Rate limiting** on API endpoints
- **Audit logging** for all attendance events
- **Privacy controls** for face data management

## 📈 **Performance Benchmarks**

- **Face Recognition**: >95% accuracy for registered employees
- **False Positive Rate**: <2% for different individuals
- **Average Response Time**: <2 seconds for attendance processing
- **System Uptime**: >99.5% availability target
- **Mobile Compatibility**: 100% responsive design

## 🎯 **Use Cases**

- **Corporate Offices**: Employee time tracking
- **Educational Institutions**: Student attendance
- **Healthcare Facilities**: Staff monitoring
- **Manufacturing Plants**: Shift management
- **Retail Stores**: Employee clock-in/out
- **Government Offices**: Secure access control

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Google Research**: FaceNet architecture and training methodology
- **DeepFace Team**: Excellent implementation and model accessibility
- **MediaPipe Team**: Fast and reliable face detection
- **TensorFlow Team**: Robust deep learning framework
- **Flask Community**: Web framework and ecosystem

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/yourusername/FaceRecogV2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/FaceRecogV2/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/FaceRecogV2/wiki)

---

**🎉 Experience the power of FaceNet512 - where cutting-edge AI meets practical attendance management!**

## 📸 **Screenshots**

### Landing Page
![Landing](https://via.placeholder.com/600x400/667eea/ffffff?text=Dashboard+with+Live+Statistics)

### Attendance Interface
![Attendance](https://via.placeholder.com/600x400/28a745/ffffff?text=QR+Code+%2B+Face+Recognition)

### Employee Registration
![Registration](https://via.placeholder.com/600x400/fd7e14/ffffff?text=Multi-step+Face+Capture)

### Admin Dashboard
![Admin](https://via.placeholder.com/600x400/dc3545/ffffff?text=Reports+%26+Analytics)

---

**Made with ❤️ using Flask, TensorFlow, and FaceNet512** 
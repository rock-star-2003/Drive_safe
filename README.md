Driver Monitoring System

A comprehensive computer vision-based system for monitoring driver alertness and attention using facial landmarks and hand detection.

Features

Core Detection Capabilities
- Drowsiness Detection: Monitors eye closure using Eye Aspect Ratio (EAR)
- Distraction Detection: Tracks head pose to detect when driver is looking away
- Hand Near Face Detection: Identifies when hands are near the face area

Alert System
- Visual Alerts: On-screen warnings with color-coded status
- Audible Alerts: Beep sounds for immediate warnings
- Voice Alerts: Text-to-speech vocal warnings for prolonged issues
- Configurable Timing: Customizable alert thresholds and intervals

User Interface
- Dual Interface Options:
  - main_final.py: Simple OpenCV-based interface
  - main_tkinter.py: Advanced Tkinter GUI with real-time metrics
- Real-time Monitoring: Live video feed with overlay information
- Feature Toggle: Enable/disable specific detection features
- System Status Panel: Comprehensive metrics display

Data Logging
- Frame-by-Frame Logging: EAR, yaw angle, and state data
- Event Logging: Timestamped records of all alert events
- CSV Export: Structured data for analysis and reporting

Installation

Prerequisites
- Python 3.7 or higher
- Webcam
- Windows, macOS, or Linux

Required Packages
pip install opencv-python mediapipe numpy pyttsx3 pillow

Optional Dependencies
- winsound (Windows only, included by default)
- tkinter (usually included with Python)

Usage

Simple Interface (main_final.py)
python main_final.py
- Press ESC or Q to exit
- Basic console output with OpenCV display

Advanced GUI (main_tkinter.py)
python main_tkinter.py
- Full graphical interface with controls
- Real-time metrics display
- Feature configuration panel
- Event logging display

Configuration

Detection Thresholds
- EAR_THRESHOLD: 0.20 (Eye Aspect Ratio for drowsiness detection)
- YAW_THRESHOLD: 20° (Head rotation angle for distraction)
- Alert Intervals: Configurable timing for repeated alerts

Customization
Modify these parameters in the code:
- Alert timing thresholds
- Beep sound frequency and duration
- Voice alert messages
- Detection sensitivity

System Requirements

Minimum
- CPU: Dual-core processor
- RAM: 4GB
- Webcam: 640x480 resolution
- OS: Windows 7+, macOS 10.12+, or Ubuntu 16.04+

Recommended
- CPU: Quad-core processor
- RAM: 8GB
- Webcam: 720p or higher
- Good lighting conditions for optimal detection

File Structure

driver-monitoring-system/
├── main_final.py          # Simple OpenCV version
├── main_tkinter.py        # Advanced GUI version
├── logs/                  # Generated log files
│   ├── driver_session_YYYYMMDD_HHMMSS.csv
│   └── driver_events_YYYYMMDD_HHMMSS.csv
└── README.md

Log Files

Frame Log (driver_session_*.csv)
- Timestamp
- Frame index
- EAR value
- Yaw angle
- Current state

Event Log (driver_events_*.csv)
- Timestamp
- Frame index
- Event type
- EAR value
- Yaw angle

Troubleshooting

Common Issues

1. Webcam not detected
   - Check camera permissions
   - Ensure no other application is using the camera
   - Try different camera index (change cv2.VideoCapture(0) to cv2.VideoCapture(1))

2. Poor detection accuracy
   - Ensure good lighting on face
   - Position camera at eye level
   - Remove glasses if causing reflections

3. Performance issues
   - Close other applications
   - Reduce camera resolution in code
   - Disable unused detection features

4. Audio alerts not working
   - Check system volume
   - Windows: Ensure winsound is available
   - Non-Windows: Terminal bell may be used

Error Messages
- "Cannot open webcam": Camera access issue
- TTS initialization errors: Text-to-speech engine unavailable
- MediaPipe errors: Check installation and dependencies

Development

Extending the System
- Add new detection algorithms
- Implement machine learning models
- Integrate with vehicle systems
- Add network connectivity for remote monitoring

Code Structure
- DriverMonitoringSystem: Core detection engine
- DriverMonitoringApp: GUI application (tkinter version)
- Modular design for easy feature addition

License

This project is intended for educational and research purposes. Please ensure compliance with local regulations when deploying in real vehicles.

Support

For issues and questions:
1. Check troubleshooting section
2. Verify all dependencies are installed
3. Ensure proper lighting and camera positioning
4. Review log files for detailed error information

Safety Notice

This system is designed as a driver assistance tool and should not replace attentive driving. Always maintain focus on the road and use this system as a supplementary safety measure.

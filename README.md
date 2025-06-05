# Drowsiness Detection System

A real-time driver drowsiness detection system with a modern web interface. The system uses YOLOv5 for drowsiness detection and provides real-time feedback through a web browser.

## Features

- Real-time video feed with drowsiness detection
- Modern, responsive web interface
- Visual and audio alerts when drowsiness is detected
- Session statistics tracking
- Easy-to-use controls

## Requirements

- Python 3.8 or higher
- Webcam
- Modern web browser

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd drowsiness-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the YOLOv5 model weights file in the correct location:
```
yolov5/runs/train/exp15/weights/last.pt
```

2. Start the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

4. Click the "Start Detection" button to begin monitoring.

## Controls

- **Start/Stop Detection**: Toggle button to start or stop the drowsiness detection
- **System Status**: Shows the current detection and alarm status
- **Statistics**: Displays session duration and number of drowsy events detected

## Notes

- The system requires a webcam to function
- Make sure you have the `alarm.wav` file in the root directory
- The web interface is optimized for modern browsers
- For best results, ensure good lighting conditions 
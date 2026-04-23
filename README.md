Abandoned Bag Detection
What this project is

This is a simple real-time system to detect abandoned bags in video footage.

It uses a custom YOLO model to detect people and bags, and then applies some basic logic to figure out whether a bag has been left unattended.

The idea is to simulate a CCTV-like use case where you want to automatically flag suspicious objects without manual monitoring.

How it works (in plain terms)
The model detects objects in each frame (person, bag, etc.)
Each object is tracked across frames
For every bag, the system checks:
Is there a person close to it?
If no person is near the bag for a certain amount of time, it is marked as abandoned
Files in this project
bag_detection/
│
├── abandoned_bag_owner_tracking.py   # Main script (run this)
├── detection.py                      # Handles YOLO detection
├── best.pt                           # Trained model weights
├── adnan.mp4                         # Sample video
├── venv/                             # Virtual environment (optional)
├── __pycache__/                      # Python cache
├── .gitignore
Requirements
Python 3.8+
torch
ultralytics
opencv-python

Install everything with:

pip install torch ultralytics opencv-python
How to run

Just run:

python abandoned_bag_owner_tracking.py

Make sure the video path inside the script is correct:

video_path = "adnan.mp4"

You can change this to:

a different video file
webcam (0)
CCTV stream
Model
best.pt is a custom-trained YOLO model
It detects:
person
bag / backpack / suitcase (depending on training)
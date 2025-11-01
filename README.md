# ITS: Intelligent Triage System (Face Recognition & Appointment Manager)
This project provides a robust solution for enhancing patient check-in and triage in a clinical setting. By combining real-time face recognition with a Tkinter-based management console, the system instantly identifies returning patients and displays their critical medical history and pending appointments.

## ✨ Key Features
Real-Time Face Recognition: Uses OpenCV and face_recognition to identify registered patients upon camera activation.

Intelligent Triage Console: Displays immediate alerts for allergies and a brief patient history, along with upcoming appointment status.

Patient Registration: GUI-based form collects patient data and captures multiple images to create a stable face encoding.

Appointment Management: Full CRUD (Create, Read, Update, Delete) capability for booking, viewing, and canceling patient appointments.

Doctor Roster: Dedicated management module for registering doctors and viewing their specialized availability and contact information.

Cyberpunk UI: Features a stylized, dark-theme console using Tkinter and custom OpenCV overlays for a distinct aesthetic.

## 💻 Technology Stack
Core Language: Python 3.x

Face Recognition: face_recognition, dlib

Image Processing: OpenCV (cv2)

GUI: Tkinter (for menus and forms)

Data Storage: .json files (for patient, doctor, and appointment records) and .pkl (for face encodings).

## ⚙️ Installation and Setup
Follow these steps to get your local copy up and running.

1. Clone the repository
Bash

git clone https://github.com/YOUR_USERNAME/ITS-System.git
cd ITS-System
2. Install Dependencies
The project relies on several key libraries. Due to the complexity of dlib (a dependency of face_recognition), it's often best to install system-level dependencies first.

On Windows/Linux/macOS:

Bash

# Recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required Python packages
pip install opencv-python numpy face-recognition
3. Initialize Data Directories
The system automatically creates these folders and files, but you can ensure they exist:

Bash

mkdir known_faces
touch patient_data.json appointments.json doctor_data.json
## ▶️ Running the Application
Execute the main Python file to launch the system's main menu:

Bash

python app.py
The application will first load all existing data and encodings, then present the ITS System Main Menu in a separate window.

## 💡 Workflow
Register a Doctor: Go to Doctor Management -> Register New Doctor. You must have doctors registered to book appointments.

Register a Patient: Go to Register New Patient. Fill out the form, then the system will open the camera to capture face data.

Start Triage: Go to Existing User (Start Triage). The live video feed will open. When a recognized patient appears, their critical data and appointments will populate the side console instantly.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

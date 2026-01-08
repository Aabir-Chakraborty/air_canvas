# Air Canvas - Listen to Your Air Writing

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-red.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

A revolutionary touchless interaction system that captures free-form air writing using hand gestures, converts it to text using OCR, and transforms it into audio output. Write in the air with your finger and listen to your writing!

## 🌟 Features

- **Touchless Air Writing**: Write characters in mid-air using your fingertip
- **Multi-Color Support**: Choose from Blue, Green, Red, and Yellow colors
- **AI-Powered OCR**: Automatic handwriting recognition using Microsoft TrOCR
- **Text-to-Speech**: Convert recognized text to natural-sounding audio
- **Save Functionality**: Capture and store your air drawings
- **Real-Time Performance**: ~30-60 FPS tracking with minimal latency
- **Hardware Independence**: Works with any standard webcam (no special devices needed)

## 🎯 Use Cases

- **Accessibility**: Communication tool for individuals with speech or mobility impairments
- **Education**: Interactive teaching and learning environments
- **Smart Environments**: Touchless control in public spaces and medical settings
- **AR/VR Applications**: Natural user interfaces for immersive experiences
- **Hygiene-Critical Settings**: Contactless interaction in healthcare and food service

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Operating System: Windows, macOS, or Linux

### Dependencies

```bash
pip install opencv-python
pip install numpy
pip install mediapipe
pip install gtts
pip install transformers
pip install torch
pip install pillow
pip install IPython
```

### Additional Requirements

For audio playback on Linux:
```bash
sudo apt-get install mpg321
```

For macOS:
```bash
brew install mpg123
```

For Windows: Audio plays automatically using the default media player.

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/air-canvas.git
cd air-canvas
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python index.py
```

4. Position yourself 2-3 feet away from the webcam

5. Start writing in the air with your index finger!

## 📖 How to Use

### Basic Controls

1. **Writing**: Point your index finger towards the camera and move it to write
2. **Pause Writing**: Make a fist to stop drawing temporarily
3. **Change Color**: Point to the color buttons at the top (Blue, Green, Red, Yellow)
4. **Clear Canvas**: Select the "CLEAR" button
5. **Save & Convert**: Click the "SAVE" button to capture, recognize text, and play audio

### Hand Gestures

- **Index Finger Extended**: Drawing mode (writes on canvas)
- **Thumb Close to Index Finger**: Pause mode (stops drawing)
- **Point at Buttons**: Select tools and colors

### Interface Elements

```
[CLEAR] [BLUE] [GREEN] [RED] [YELLOW] [SAVE]
```

- **CLEAR**: Erases all drawings
- **Color Buttons**: Changes drawing color
- **SAVE**: Triggers OCR and text-to-speech conversion

## 🔬 Technical Architecture

### Phase I: Air Writing Capture
- **Technology**: MediaPipe Hands for 21-point hand landmark detection
- **Accuracy**: >95% fingertip tracking
- **Performance**: 30-60 FPS real-time processing
- **Output**: Trajectory visualization on canvas

### Phase II: Text Recognition
- **Model**: Microsoft TrOCR (Transformer-based OCR)
- **Approach**: End-to-end handwriting recognition
- **Features**: Handles variations in handwriting styles
- **Output**: Extracted text from air-drawn strokes

### Phase III: Audio Conversion
- **Technology**: Google Text-to-Speech (gTTS)
- **Processing**: Natural language generation with proper intonation
- **Output**: Synthesized speech audio (MP3 format)

### System Flow

```
Webcam Input → Hand Detection → Trajectory Tracking → Canvas Drawing
     ↓
Screenshot Capture → OCR Processing → Text Extraction
     ↓
Text-to-Speech → Audio Generation → Playback
```

## 📊 Performance Metrics

- **Hand Tracking Accuracy**: >95%
- **Frame Rate**: 30-60 FPS
- **OCR Accuracy**: 92-97% (comparable systems)
- **Latency**: 300-500ms (trajectory → OCR → speech)
- **Operating Distance**: 2-3 feet from camera
- **Resolution**: 640×480 pixels

## 🔧 Configuration

### Adjusting Detection Sensitivity

In `index.py`, modify the MediaPipe initialization:

```python
hands = mpHands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7,  # Adjust between 0.5-0.9
    min_tracking_confidence=0.5     # Add if needed
)
```

### Changing Drawing Parameters

```python
# Line thickness
cv2.line(paintWindow, point1, point2, color, 2)  # Change 2 to desired thickness

# Buffer size for smoother lines
bpoints = [deque(maxlen=1024)]  # Increase for longer strokes
```

### Custom Voice Settings

```python
# Change language
language = 'en'  # 'es' for Spanish, 'fr' for French, etc.

# Adjust speech speed
tts = gTTS(text=ocr_result, lang=language, slow=True)  # slow=True for slower speech
```

## 📁 Project Structure

```
air-canvas/
│
├── index.py                 # Main application file
├── screenshots/             # Saved air writing images
│   └── screenshot.png
├── output.mp3              # Generated audio file
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

- [ ] Reduce real-time lag in text extraction
- [ ] Support for multiple languages
- [ ] Enhanced OCR accuracy for cursive writing
- [ ] Mobile application development
- [ ] Multi-user collaboration features
- [ ] Custom gesture commands

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Known Issues & Future Work

- **Current Limitations**:
  - Lag in real-time text extraction and audio conversion
  - Lighting conditions affect tracking accuracy
  - Limited to English language recognition

- **Future Enhancements**:
  - Real-time OCR processing
  - Multi-language support
  - Gesture-based commands
  - Cloud integration for model optimization
  - Mobile app version

## 📚 Research & References

This project is based on research conducted at Thapar Institute of Engineering and Technology. For detailed methodology and results, refer to the research paper:

**"Listen to Air Canvas"** by Aabir Chakraborty and Ashu Sharma

Key achievements:
- 96.7% recognition accuracy in person-dependent evaluations
- Hardware-independent solution
- Seamless gesture-to-audio pipeline

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Aabir Chakraborty** - Thapar Institute of Engineering and Technology
- **Ashu Sharma** - sharma.ashu204@gmail.com

## 🙏 Acknowledgments

- MediaPipe team for the hand tracking solution
- Microsoft for the TrOCR model
- Google for the Text-to-Speech API
- OpenCV community

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Email: sharma.ashu204@gmail.com

## 🌐 Links

- [Research Paper](link-to-paper)
- [Demo Video](link-to-demo)
- [Documentation](link-to-docs)

---

**Made with ❤️ for accessible and inclusive technology**

*Write freely, listen easily!*
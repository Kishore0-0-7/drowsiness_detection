# Drowsiness Detection System

![Drowsiness Detection](https://img.shields.io/badge/Safety-Drowsiness%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

A real-time drowsiness detection system designed to monitor driver alertness and prevent accidents caused by fatigue. The application uses computer vision and machine learning techniques to detect signs of drowsiness by analyzing eye closure patterns and facial expressions.

## Features

- **Real-time drowsiness detection** using Eye Aspect Ratio (EAR) analysis
- **Facial emotion recognition** to detect driver mood and alertness level
- **Visual and audible alerts** when drowsiness is detected
- **Comprehensive dashboard** with statistics and event tracking
- **Data visualization** for analysis of drowsiness patterns
- **Customizable settings** for detection sensitivity
- **Report generation** for analysis and record-keeping
- **Export/Import functionality** for settings and data

## Screenshots

<p align="center">
  <em>Screenshots will be added here</em>
</p>

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

1. Launch the application using the command above
2. Position yourself in front of the camera
3. Adjust the detection settings if needed
4. The system will automatically start monitoring for signs of drowsiness
5. When drowsiness is detected, an alarm will sound to alert you

### Settings Configuration

- **EAR Threshold**: Adjust the Eye Aspect Ratio threshold (lower values increase sensitivity)
- **Time Threshold**: Set how long eyes must be closed to trigger an alert
- **Alarm Sound**: Choose from different alarm sounds
- **UI Theme**: Select your preferred interface style

## How It Works

The system uses several key technologies:

1. **MediaPipe Face Mesh**: For detecting facial landmarks
2. **Eye Aspect Ratio (EAR)**: A measure of eye openness calculated from eye landmarks
3. **Emotion Recognition**: Neural network-based analysis of facial expressions
4. **Streamlit**: For the interactive user interface

When the EAR falls below the threshold for a specified duration, the system identifies it as drowsiness and triggers an alert.

## Advanced Features

### Reporting and Analytics

The system maintains a log of drowsiness events and can generate detailed reports including:

- Frequency of drowsiness episodes
- Distribution of events by time of day
- Correlation with detected emotions
- Statistics on eye closure patterns

### Data Export/Import

- Export all settings and event data as JSON
- Import settings from previously saved configurations
- Export events to CSV for external analysis

## Troubleshooting

- **Camera not detected**: Ensure your webcam is properly connected and not in use by another application
- **Face not detected**: Adjust lighting conditions and position yourself directly in front of the camera
- **False alarms**: Increase the EAR threshold or time threshold to reduce sensitivity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Eye Aspect Ratio calculation is based on research by Soukupová and Čech
- Emotion recognition uses the facial-emotion-recognition library
- Face mesh detection powered by Google's MediaPipe

## Contact

For support or inquiries, please contact [your-email@example.com](mailto:your-email@example.com).

---

*Stay alert, stay safe!*

# HandSignAI-
HandSignAI – Real-Time Sign Language to Text Converter <br>

HandSignAI is an AI-powered real-time application that translates sign language gestures into text using computer vision and deep learning techniques. The project aims to bridge the communication gap between the hearing-impaired community and others by providing an accessible, fast, and accurate solution for gesture recognition. <br>

# Features <br>
1)Real-Time Detection: Converts hand signs into text instantly using a live webcam feed. <br>
2)Deep Learning Model: Built with CNN / Transfer Learning for accurate gesture recognition. <br>
3)Sign Language Support: Recognizes alphabets (A–Z), numbers, and commonly used gestures.<br>
4)User-Friendly Interface: Displays detected signs and corresponding text on screen.<br>
5)Lightweight and Fast: Optimized for real-time performance on standard hardware.<br>
6)Accessibility Focused: Helps improve inclusivity by enabling seamless communication.<br>

# Tech Stack
Python <br>
OpenCV – For hand tracking and real-time image processing <br>
TensorFlow / Keras – For training and deploying the gesture recognition model <br>
NumPy and Pandas – Data preprocessing and handling <br>
Matplotlib / Seaborn – Visualization of training results <br>
Streamlit / Tkinter (optional) – For building a user interface <br>

# How It Works <br>
Capture – The webcam captures the user’s hand signs. <br>
Preprocess – OpenCV isolates and processes the hand region. <br>
Predict – The trained deep learning model classifies the gesture. <br>
Translate – The recognized sign is displayed as text in real-time. <br>

# Use Cases
Assistive Communication – Helps people with hearing or speech impairments communicate more easily. <br>
Education – Useful for learning sign language interactively. <br>
Research – Can be extended for multilingual sign language datasets. <br>
Integration – Potential integration with chat apps, mobile apps, or IoT devices for accessibility. <br>

# Future Enhancements
Add support for full words and sentences (not just alphabets). <br>
Implement voice output (text-to-speech) for detected signs. <br>
Expand to multiple sign languages (e.g., ASL, ISL, BSL). <br>
Improve accuracy with transformer-based or vision-language models.<br>

<h3> INSTALLITATION : </h3> <br>

1)Clone this repository: <br>
git clone https://github.com/SSonu16/HandSignAI.git <br>
cd HandSignAI <br>

2) Create a virtual environment : <br>
python -m venv venv <br>
source venv/bin/activate   # On Linux/Mac <br>
venv\Scripts\activate      # On Windows <br>

3)Install dependencies: <br>
  pip install -r requirements.txt <br>
  pip install opencv-python mediapipe scikit-learn joblib numpy <br>
  python sign_language_to_text_starter.py collect <br>

4)Ensure you have a working webcam connected to your system. <br>

5)python sign_language_to_text_starter.py train <br>
  python sign_language_to_text_starter.py run <br>

<h3> Usage </h3> <br>

1) Run the application: <br>
python app.py <br>
2)The webcam will start and detect hand signs in real time. <br>
3)The predicted sign will be displayed as text on the screen. <br>

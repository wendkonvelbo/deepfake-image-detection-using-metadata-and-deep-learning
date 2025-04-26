# deepfake-image-detection-using-metadata-and-deep-learning

This project focuses on detecting deepfake images by combining metadata analysis and deep learning (CNN and frequency domain techniques).
The aim is to classify real vs fake faces based on both pixel-level and metadata-level features for improved detection accuracy.

🚀 Project Structure
deepfake-image-detection/
├── README.md
├── requirements.txt
├── metadata_analysis/
│   └── metadata_preprocessing.py
├── model/
│   ├── cnn_model.py
│   ├── fft_analysis.py
├── notebook/
│   └── deepfake_detection.ipynb
├── data/ (not included in repo)
│   ├── faces_224/
│   └── metadata.csv
└── results/
    └── output_examples/

🛠️ Features
Metadata Extraction: Analyze EXIF and other metadata to spot anomalies.

Deep Learning Model: CNN-based classifier trained on RGB images.

Frequency Domain Analysis: FFT (Fast Fourier Transform) applied to detect high-frequency artifacts.

Combined Approach: Improve detection accuracy by combining visual and metadata features.

Visualization: EDA (Exploratory Data Analysis) on both images and metadata.

📦 Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/wendkonvelbo/deepfake-image-detection.git
cd deepfake-image-detection
Create a virtual environment (recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🧠 Model Training
Train the CNN model using RGB images:

bash
Copy
Edit
python model/cnn_model.py
Apply FFT preprocessing and train a second model:

bash
Copy
Edit
python model/fft_analysis.py


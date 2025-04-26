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

RESULT 
Image 1: The Pie Chart
Imagine you have a box of 100 photos. This pie chart shows that 83 of those photos are fake (red piece) and only 17 are real (purple piece). The computer needs to learn which is which, but this chart shows it will see many more fake photos than real ones during training.
Images 2 & 3: The Mountain Charts
These show how big the photos are. Most photos are about the same size - not too big and not too small. Like if you had a bunch of school photos, they'd mostly be similar sizes. The computer works better when all the photos are similar sizes.
Images 4 & 5: The Box Charts
These are like measuring all the photos with a ruler. The red box shows where most photo sizes fall. The little circles far away are the unusually big or small photos. This helps us know if some photos need special attention because they're different.
Images 6 & 7: The Shape Charts
These funny-looking shapes show us where most photo sizes are. The fatter parts mean there are lots of photos that size. It looks like a sound wave because there are several common sizes in the collection.
Image 8: Face Examples
This is just showing us examples of faces in the photos. Some are clear, some are blurry. The computer needs to see lots of different types of faces to learn well.
Image 9: Real vs. Fake Faces
This is the most important picture! It shows faces labeled as either "REAL" or "FAKE." The amazing thing is that many fake faces look so real that it's hard for people to tell the difference. The computer has to learn tiny details that we might miss - like weird shadows, strange eye reflections, or skin that looks too smooth.
The computer is like a detective looking for clues that are so small that most people wouldn't notice them. It can find patterns in the photos that help it figure out which faces were created by computers (fake) and which are photos of real people.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cv2
import os
import zipfile
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torchvision.models as models
from sklearn.model_selection import train_test_split
from torchvision import transforms
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Set paths to your files
dataset_path = "D:/downloads/metadata.csv.zip"
image_path = "D:/downloads/faces_224.zip"

# Load dataset
df = pd.read_csv(dataset_path)

# Display dataset summary
print(df.head())
print(df.tail())
print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print("Duplicated Rows:", df.duplicated().sum())
print("Missing Values:\n", df.isnull().sum())
print(df.info())
print("Unique Values Per Column:\n", df.nunique())

# Classifying Features
def classify_features(df):
    categorical_features = []
    non_categorical_features = []
    discrete_features = []
    continuous_features = []

    for column in df.columns:
        if df[column].dtype == 'object':
            if df[column].nunique() < 10:
                categorical_features.append(column)
            else:
                non_categorical_features.append(column)
        elif df[column].dtype in ['int64', 'float64']:
            if df[column].nunique() < 10:
                discrete_features.append(column)
            else:
                continuous_features.append(column)

    return categorical_features, non_categorical_features, discrete_features, continuous_features

categorical, non_categorical, discrete, continuous = classify_features(df)
print("Categorical Features:", categorical)
print("Non-Categorical Features:", non_categorical)
print("Discrete Features:", discrete)
print("Continuous Features:", continuous)

df.fillna("Not Available", inplace=True)

for col in categorical:
    print(f"{col}: {df[col].unique()}\n")
    print(df[col].value_counts())
    print()

# Define a custom color palette with pink and black
pink_black_palette = ['#FFC0CB', '#000000']  # Pink and Black

# Apply the custom palette to your count plots
for col in categorical:
    plt.figure(figsize=(15, 6))
    sns.countplot(x=df[col], palette=pink_black_palette)
    plt.title(f"Distribution of {col}")
    plt.show()



# Define a custom red and purple color palette
red_purple_palette = ['#FF0000', '#8B008B', '#FF6347', '#800080', '#FF4500']  # Red to Purple gradient

# Loop through each categorical column
for col in categorical:
    plt.figure(figsize=(10, 7))
    # Create the pie chart with the custom red and purple palette
    plt.pie(df[col].value_counts(), labels=df[col].value_counts().index, autopct='%1.1f%%', colors=red_purple_palette[:len(df[col].value_counts())], textprops={'fontsize': 12})
    plt.title(f"Proportion of {col}")
    plt.show()


# Distribution plots for numerical features
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=90)
    plt.show()

# Boxplots
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.boxplot(x=df[col], palette='hls')
    plt.title(f"Boxplot of {col}")
    plt.xticks(rotation=90)
    plt.show()

# Violin plots
for col in discrete + continuous:
    plt.figure(figsize=(12,5))
    sns.violinplot(x=df[col], palette='hls')
    plt.title(f"Violin Plot of {col}")
    plt.xticks(rotation=90)
    plt.show()

# Balancing data
real_df = df[df["label"] == "REAL"]
fake_df = df[df["label"] == "FAKE"]
sample_size = min(len(real_df), len(fake_df))
real_df = real_df.sample(sample_size, random_state=42)
fake_df = fake_df.sample(sample_size, random_state=42)
sample_meta = pd.concat([real_df, fake_df])

# Train, val, test split
Train_set, Test_set = train_test_split(sample_meta, test_size=0.2, random_state=42, stratify=sample_meta['label'])
Train_set, Val_set  = train_test_split(Train_set, test_size=0.3, random_state=42, stratify=Train_set['label'])
print("Train, Validation, and Test Set Sizes:", Train_set.shape, Val_set.shape, Test_set.shape)

# Image processing
if os.path.isfile(image_path):
    try:
        with zipfile.ZipFile(image_path, 'r') as zip_ref:
            image_files = sorted(zip_ref.namelist())
            selected_images = image_files[:9]
    except zipfile.BadZipFile as e:
        print(f"Error opening zip file: {e}")
else:
    print(f"The file '{image_path}' does not exist. Please check the path.")

# Show 9 images
plt.figure(figsize=(10, 10))
with zipfile.ZipFile(image_path, 'r') as zip_ref:
    for index, image_file in enumerate(selected_images):
        with zip_ref.open(image_file) as img_file:
            img_data = img_file.read()
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

        if image is not None:
            plt.subplot(3, 3, index + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Image {index + 1}')
            plt.axis('off')
        else:
            print(f"Error reading image: {image_file}")
plt.show()

# Print resolution of first 10 images
with zipfile.ZipFile(image_path, 'r') as zip_ref:
    for i, image_file in enumerate(image_files[:10]):
        with zip_ref.open(image_file) as img_file:
            img_data = img_file.read()
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            height, width, _ = image.shape
            print(f"Resolution of image {i+1}: {width} x {height}")
        else:
            print(f"Error reading image {i+1}")

# Plot sample images from training set
plt.figure(figsize=(15,15))
with zipfile.ZipFile(image_path, 'r') as zip_ref:
    for cur, i in enumerate(Train_set.index[25:50]):
        image_file = Train_set.loc[i,'videoname'][:-4] + '.jpg'
        if image_file in zip_ref.namelist():
            with zip_ref.open(image_file) as img_file:
                img_data = img_file.read()
                image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

            if image is not None:
                plt.subplot(5, 5, cur+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.xlabel('FAKE Image' if Train_set.loc[i,'label']=='FAKE' else 'REAL Image')
            else:
                print(f"Error reading image: {image_file}")
        else:
            print(f"Image file not found in zip archive: {image_file}")
plt.show()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, zip_path, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.zip_path = zip_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        image_file = row['videoname'].replace('.mp4', '.jpg')
        label = 1 if row['label'] == 'FAKE' else 0

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open(image_file) as img_file:
                img_data = img_file.read()
                image = Image.open(io.BytesIO(img_data)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

train_dataset = DeepfakeDataset(Train_set, image_path, transform=transform)
val_dataset = DeepfakeDataset(Val_set, image_path, transform=transform)
test_dataset = DeepfakeDataset(Test_set, image_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Frequency domain visualization
import numpy.fft as fft
plt.figure(figsize=(15, 5))
with zipfile.ZipFile(image_path, 'r') as zip_ref:
    for i in range(3):
        image_file = selected_images[i]
        with zip_ref.open(image_file) as img_file:
            img_data = img_file.read()
            image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)

        f = fft.fft2(image)
        fshift = fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        plt.subplot(2, 3, i+1)
        plt.imshow(image, cmap='gray')
        plt.title("Original")

        plt.subplot(2, 3, i+4)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title("FFT Magnitude")
plt.tight_layout()
plt.show()

# Predictions and FFT Visualization from CNN
model.eval()
class_names = ['REAL', 'FAKE']
plt.figure(figsize=(15, 10))

with torch.no_grad():
    with zipfile.ZipFile(image_path, 'r') as zip_ref:
        shown = 0
        for idx, (img_tensor, label) in enumerate(test_loader):
            if shown >= 5:
                break

            img_tensor = img_tensor.to(device)
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

            file_index = Test_set.index[idx]
            video_name = Test_set.loc[file_index, 'videoname']
            image_file = video_name.replace('.mp4', '.jpg')

            if image_file not in zip_ref.namelist():
                continue

            with zip_ref.open(image_file) as img_file:
                img_data = img_file.read()
                original_img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

            f = fft.fft2(gray_img)
            fshift = fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

            plt.subplot(5, 3, shown * 3 + 1)
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Image\nTrue: {class_names[label.item()]}, Pred: {class_names[predicted.item()]}")
            plt.axis('off')

            plt.subplot(5, 3, shown * 3 + 2)
            plt.imshow(gray_img, cmap='gray')
            plt.title("Grayscale")
            plt.axis('off')

            plt.subplot(5, 3, shown * 3 + 3)
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title("FFT Spectrum")
            plt.axis('off')

            shown += 1

plt.tight_layout()
plt.show()
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
data_dir = "d:/downloads/deepfake/faces_224"  # adjust if needed

# Load your already split DataFrames
# train_df, val_df, test_df assumed to be already defined
from sklearn.model_selection import train_test_split

# Assuming `df` is your cleaned dataframe
train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.3, stratify=train_val_df['label'], random_state=42)
# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Custom Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.root_dir, row['videoname'].replace('.mp4', '.jpg'))
        image = Image.open(img_path).convert('RGB')
        label = 1 if row['label'] == 'FAKE' else 0

        if self.transform:
            image = self.transform(image)
        return image, label

# Create datasets
train_dataset = DeepfakeDataset(train_df, data_dir, transform)
val_dataset = DeepfakeDataset(val_df, data_dir, transform)
test_dataset = DeepfakeDataset(test_df, data_dir, transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# Define CNN model (based on a pre-trained model for better accuracy)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

# Evaluation on test set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['REAL', 'FAKE']))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
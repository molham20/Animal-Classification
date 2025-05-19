import os
import cv2
import numpy as np
import pickle
import gradio as gr
from PIL import Image
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# -------------------- CONFIG --------------------
class_names = ['Buffalo', 'cat', 'deer', 'dog', 'Elephant', 'horse', 'lion', 'Rhino', 'Zebra']
DATA_DIR = "Data"
CNN_MODEL_PATH = "cnn_model.h5"
KNN_MODEL_PATH = "knn_model.pkl"
K = 5

# -------------------- ENHANCEMENT --------------------
def enhance_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# -------------------- CNN --------------------
def train_and_save_cnn():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping # type: ignore
    from tensorflow.keras import Sequential # type: ignore
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore

    X = []
    y = []
    print("Training CNN model...")

    for idx, class_name in enumerate(class_names):
        folder = os.path.join(DATA_DIR, class_name)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = enhance_image(img)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            X.append(img)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_test, y_test),
              epochs=20,
              callbacks=[early_stop])

    model.save(CNN_MODEL_PATH)
    print(f"Saved trained CNN model to {CNN_MODEL_PATH}")
    return model

def load_cnn_model():
    if os.path.exists(CNN_MODEL_PATH):
        return load_model(CNN_MODEL_PATH)
    else:
        return train_and_save_cnn()

# -------------------- KNN --------------------
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.le = LabelEncoder()

    def fit(self, X, y):
        self.X_train = X
        self.y_train = self.le.fit_transform(y)
        self.mean = np.mean(self.X_train, axis=0)
        self.std = np.std(self.X_train, axis=0)
        self.X_train = (self.X_train - self.mean) / (self.std + 1e-8)

    def predict(self, X_test):
        X_test = (X_test - self.mean) / (self.std + 1e-8)
        predictions = []
        for x_test in tqdm(X_test, desc="Predicting"):
            distances = 1 - np.dot(self.X_train, x_test) / (
                np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x_test) + 1e-8)
            k_indices = np.argpartition(distances, self.k)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return self.le.inverse_transform(predictions)

cnn_feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img = preprocess_input(np.expand_dims(img_array.astype(np.float32), axis=0))
    features = cnn_feature_extractor.predict(img, verbose=0)
    return features.flatten()

def save_knn_model(model, X, y):
    with open(KNN_MODEL_PATH, "wb") as f:
        pickle.dump((model, X, y), f)

def load_knn_model():
    if os.path.exists(KNN_MODEL_PATH):
        with open(KNN_MODEL_PATH, "rb") as f:
            model, X, y = pickle.load(f)
            model.fit(X, y)
            return model
    else:
        print("Training KNN model...")
        X, y = [], []
        for label in class_names:
            folder = os.path.join(DATA_DIR, label)
            for filename in os.listdir(folder):
                path = os.path.join(folder, filename)
                try:
                    img = cv2.imread(path)
                    img = enhance_image(img)
                    feat = extract_features(img)
                    X.append(feat)
                    y.append(label)
                except:
                    continue
        X = np.array(X)
        y = np.array(y)
        model = KNNClassifier(k=K)
        model.fit(X, y)
        save_knn_model(model, X, y)
        return model

# -------------------- GUI --------------------
cnn_model = load_cnn_model()
knn_model = load_knn_model()

def predict(image, model_type):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = enhance_image(image)

    if model_type == "CNN":
        img_resized = cv2.resize(image, (128, 128)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)
        preds = cnn_model.predict(img_input)[0]
        return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

    elif model_type == "KNN":
        features = extract_features(image).reshape(1, -1)
        pred_label = knn_model.predict(features)[0]
        return {label: (1.0 if label == pred_label else 0.0) for label in class_names}

    else:
        return {"Error": 1.0}

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload an Animal Image"),
        gr.Radio(choices=["CNN", "KNN"], label="Select Model", value="CNN")
    ],
    outputs=gr.Label(num_top_classes=3),
    title="Animal Classifier with CNN & KNN",
    description="Upload an animal image and choose the model for prediction."
)

iface.launch()

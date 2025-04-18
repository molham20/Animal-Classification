import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn import svm, ensemble, neural_network
from sklearn.metrics import classification_report
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess # type: ignore
import gradio as gr
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# --- CONFIG ---
IMG_SIZE = (64, 64)
DATA_DIR = "animals/train"  # should contain subfolders like "cat", "dog", etc.
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Check1")

# --- LOAD IMAGES ---
def load_data(data_dir):
    X, y = [], []
    class_names = os.listdir(data_dir)
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                X.append(np.array(img))
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y), class_names

print("Check2")


X, y, class_names = load_data(DATA_DIR)
X_flat = X.reshape(len(X), -1) / 255.0  # for ML models
X_norm = X / 255.0  # for DL models

X_train_f, X_test_f, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
X_train_d, X_test_d, _, _ = train_test_split(X_norm, y, test_size=0.2, random_state=42)

print("Check3")


# --- FAST SVM ---
svm_model = LinearSVC()
svm_model.fit(X_train_f, y_train)
svm_preds = svm_model.predict(X_test_f)
print("✅ LinearSVC Accuracy:", accuracy_score(y_test, svm_preds))
joblib.dump(svm_model, f"{MODEL_DIR}/svm.pkl")
svm_preds = svm_model.predict(X_test_f)
print("SVM Accuracy:", accuracy_score(y_test, svm_preds))


# --- Random Forest ---
rf_model = ensemble.RandomForestClassifier()
rf_model.fit(X_train_f, y_train)
joblib.dump(rf_model, f"{MODEL_DIR}/rf.pkl")
rf_preds = rf_model.predict(X_test_f)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))


# --- MLP ---
mlp_model = neural_network.MLPClassifier(max_iter=300)
mlp_model.fit(X_train_f, y_train)
joblib.dump(mlp_model, f"{MODEL_DIR}/mlp.pkl")
mlp_preds = mlp_model.predict(X_test_f)
print("MLP Accuracy:", accuracy_score(y_test, mlp_preds))


# --- CNN ---
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_names), activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_d, y_train, validation_data=(X_test_d, y_test), epochs=5)
cnn_model.save(f"{MODEL_DIR}/cnn.h5")
cnn_loss, cnn_acc = cnn_model.evaluate(X_test_d, y_test, verbose=0)
print("CNN Accuracy:", cnn_acc)


# --- ResNet50 ---
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3), pooling='avg')
resnet_model = Sequential([
    resnet_base,
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
resnet_model.fit(resnet_preprocess(X_train_d), y_train, validation_data=(resnet_preprocess(X_test_d), y_test), epochs=5)
resnet_model.save(f"{MODEL_DIR}/resnet50.h5")
resnet_loss, resnet_acc = resnet_model.evaluate(resnet_preprocess(X_test_d), y_test, verbose=0)
print("ResNet50 Accuracy:", resnet_acc)


# --- PREDICTION FUNCTION ---
def preprocess(image, flatten=False, use_resnet=False):
    img = image.resize(IMG_SIZE).convert('RGB')
    img_array = np.array(img)
    if flatten:
        return img_array.reshape(1, -1) / 255.0
    elif use_resnet:
        return resnet_preprocess(np.expand_dims(img_array, axis=0))
    else:
        return np.expand_dims(img_array / 255.0, axis=0)

def predict(image, model_choice):
    if model_choice == "SVM":
        model = joblib.load(f"{MODEL_DIR}/svm.pkl")
        processed = preprocess(image, flatten=True)
        pred = model.predict(processed)[0]
    elif model_choice == "Random Forest":
        model = joblib.load(f"{MODEL_DIR}/rf.pkl")
        processed = preprocess(image, flatten=True)
        pred = model.predict(processed)[0]
    elif model_choice == "MLP":
        model = joblib.load(f"{MODEL_DIR}/mlp.pkl")
        processed = preprocess(image, flatten=True)
        pred = model.predict(processed)[0]
    elif model_choice == "CNN":
        model = tf.keras.models.load_model(f"{MODEL_DIR}/cnn.h5")
        processed = preprocess(image)
        pred = np.argmax(model.predict(processed), axis=1)[0]
    elif model_choice == "ResNet50":
        model = tf.keras.models.load_model(f"{MODEL_DIR}/resnet50.h5")
        processed = preprocess(image, use_resnet=True)
        pred = np.argmax(model.predict(processed), axis=1)[0]
    return f"Prediction: {class_names[pred]}"

# --- GRADIO GUI ---
gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Animal Image"),
        gr.Dropdown(["SVM", "CNN", "ResNet50", "Random Forest", "MLP"], label="Choose Model")
    ],
    outputs="text",
    title="Animal Classifier"
).launch()

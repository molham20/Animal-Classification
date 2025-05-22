import os
import random
import cv2
import joblib
import numpy as np
import pickle
import gradio as gr
from PIL import Image
from collections import Counter
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.models import ResNet18_Weights
from torchvision import datasets, transforms, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras import Sequential, regularizers# type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization# type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model# type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input# type: ignore
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- Configuration ----------------
class_names = ['Buffalo', 'cat', 'deer', 'dog', 'Elephant', 'horse', 'lion', 'Rhino', 'Zebra']
DATA_DIR = "Data"
CNN_MODEL_PATH = "cnn_model.h5"
KNN_MODEL_PATH = "knn_model.pkl"
SVM_MODEL_PATH = "svm_model.joblib"
RESNET_MODEL_PATH = "resnet_model.pth"
RF_MODEL_PATH = "animal_classifier_rf.joblib"
AUGMENTED_DIR = "Augmented_Data"
IMG_SIZE = 128

# -------------------- Utilities --------------------
def enhance_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# -------------------- CNN --------------------
def train_and_save_cnn():
    X, y = [], []
    print("Training CNN model...")
    for idx, cn in enumerate(class_names):
        folder = os.path.join(DATA_DIR, cn)
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None: continue
            img = enhance_image(img)
            img = cv2.resize(img, (128, 128)) / 255.0
            X.append(img); y.append(idx)
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=38, stratify=y)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    model = Sequential([
        Conv2D(32,(3,3),activation='relu',padding='same',
               kernel_regularizer=regularizers.l2(0.001), input_shape=(128,128,3)),
        BatchNormalization(),
        Conv2D(32,(3,3),activation='relu',padding='same'), BatchNormalization(),
        MaxPooling2D((2,2)), Dropout(0.25),
        Conv2D(64,(3,3),activation='relu',padding='same',
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(), Conv2D(64,(3,3),activation='relu',padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)), Dropout(0.35),
        Conv2D(128,(3,3),activation='relu',padding='same',
               kernel_regularizer=regularizers.l2(0.001)),
        BatchNormalization(), Conv2D(128,(3,3),activation='relu',padding='same'),
        BatchNormalization(), MaxPooling2D((2,2)), Dropout(0.5),
        Flatten(), Dense(256,activation='relu'), BatchNormalization(), Dropout(0.5),
        Dense(len(class_names),activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    model.fit(
        datagen.flow(X_train,y_train,batch_size=32),
        validation_data=(X_test,y_test), epochs=50,
        callbacks=[early_stop, reduce_lr]
    )
    loss, acc = model.evaluate(X_test,y_test)
    print(f"CNN test accuracy: {acc:.4f}")
    model.save(CNN_MODEL_PATH)
    return model

def load_cnn_model():
    return load_model(CNN_MODEL_PATH) if os.path.exists(CNN_MODEL_PATH) else train_and_save_cnn()

# -------------------- KNN --------------------
class KNNClassifier:
    def __init__(self, k=5): self.k=k; self.le=LabelEncoder()
    def fit(self,X,y):
        self.X_train=np.array(X); self.y_train=self.le.fit_transform(y)
        self.mean,self.std = self.X_train.mean(0), self.X_train.std(0)
        self.X_train=(self.X_train-self.mean)/(self.std+1e-8)
    def predict(self,X_test):
        X=(X_test-self.mean)/(self.std+1e-8)
        preds=[]
        for x in tqdm(X,desc="KNN predicting"):
            d = 1 - np.dot(self.X_train,x)/(np.linalg.norm(self.X_train,axis=1)*np.linalg.norm(x)+1e-8)
            idxs = np.argpartition(d,self.k)[:self.k]
            lbls = self.y_train[idxs]
            preds.append(Counter(lbls).most_common(1)[0][0])
        return self.le.inverse_transform(preds)

cnn_feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')
def extract_features(img):
    img = cv2.resize(img, (224, 224))

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    inp = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    feat = cnn_feature_extractor.predict(inp, verbose=0)
    return feat.flatten()

def load_knn_model(k=5):
    if os.path.exists(KNN_MODEL_PATH):
        model,X,y = pickle.load(open(KNN_MODEL_PATH,'rb'))
        model.fit(X,y)
        return model
    X,y=[],[]
    print("Training KNN model...")
    for cn in class_names:
        for fn in os.listdir(os.path.join(DATA_DIR,cn)):
            img=cv2.imread(os.path.join(DATA_DIR,cn,fn))
            if img is None: continue
            img=enhance_image(img)
            X.append(extract_features(img)); y.append(cn)
    knn=KNNClassifier(k=k)
    knn.fit(X,y)
    pickle.dump((knn,X,y), open(KNN_MODEL_PATH,'wb'))
    return knn

# -------------------- SVM --------------------
def load_svm_model():
    if os.path.exists(SVM_MODEL_PATH):
        return joblib.load(SVM_MODEL_PATH)
    print("Training SVM model...")
    X,y=[],[]
    for cn in class_names:
        for fn in os.listdir(os.path.join(DATA_DIR,cn)):
            img=cv2.imread(os.path.join(DATA_DIR,cn,fn))
            if img is None: continue
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(128,128))/255.0
            X.append(img.flatten()); y.append(cn)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler=StandardScaler().fit(X_train)
    svm=LinearSVC(C=1.0, max_iter=10000, dual=False, random_state=42)
    svm.fit(scaler.transform(X_train),y_train)
    joblib.dump((svm,scaler),SVM_MODEL_PATH)
    return svm, scaler

# -------------------- ResNet --------------------
def load_resnet_model():
    if os.path.exists(RESNET_MODEL_PATH):
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features,len(class_names))
        model.load_state_dict(torch.load(RESNET_MODEL_PATH))
        return model
    print("Training ResNet model...")
    tr = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = datasets.ImageFolder(DATA_DIR,transform=tr)
    train_len = int(0.8*len(dataset))
    train_ds,val_ds = random_split(dataset,[train_len,len(dataset)-train_len])
    tl = DataLoader(train_ds,batch_size=32,shuffle=True)
    vl = DataLoader(val_ds,batch_size=32)
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features,len(class_names))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    crit=nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for xb,yb in tl:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
    torch.save(model.state_dict(),RESNET_MODEL_PATH)
    return model

# -------------------- Random Forest (New Implementation) --------------------
def apply_augmentation(img):
    augs = [img, cv2.flip(img, 1)]
    for angle in (-15, 15):
        M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
        augs.append(cv2.warpAffine(img, M, (128, 128)))
    for beta in (-30, 30): 
        augs.append(cv2.convertScaleAbs(img, alpha=1.0, beta=beta))
    augs.append(cv2.GaussianBlur(img, (5, 5), 0))
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    augs.append(cv2.add(img, noise))
    crop = random.randint(0, 25)
    c = img[crop:crop+102, crop:crop+102]
    augs.append(cv2.resize(c, (128, 128)))
    return augs[:5]

def create_augmented_dataset():
    if os.path.exists(AUGMENTED_DIR):
        complete = all(os.path.isdir(os.path.join(AUGMENTED_DIR, cn)) and os.listdir(os.path.join(AUGMENTED_DIR, cn))
                       for cn in class_names)
        if complete: return
        import shutil; shutil.rmtree(AUGMENTED_DIR)
    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    for cn in class_names:
        os.makedirs(os.path.join(AUGMENTED_DIR, cn), exist_ok=True)
        for fn in os.listdir(os.path.join(DATA_DIR, cn)):
            path = os.path.join(DATA_DIR, cn, fn)
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(AUGMENTED_DIR, cn, fn), gray)
            for i, aug in enumerate(apply_augmentation(gray)):
                out = f"{os.path.splitext(fn)[0]}_aug{i}.jpg"
                cv2.imwrite(os.path.join(AUGMENTED_DIR, cn, out), aug)

def extract_rf_features(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(img)
    
    denoised = cv2.fastNlMeansDenoising(equalized, None, h=10)
    
    features, _ = hog(
        denoised,
        orientations=12,
        pixels_per_cell=(12, 12),
        cells_per_block=(3, 3),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    
    hist = cv2.calcHist([denoised], [0], None, [32], [0, 256])
    hist = hist.flatten()
    
    lbp = local_binary_pattern(denoised, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 32))
    
    return np.concatenate((features, hist, lbp_hist))

def load_rf_data():
    data = []
    labels = []
    originals = []
    
    print("🔍 Loading augmented data...")
    
    for idx, category in enumerate(class_names):
        category_path = os.path.join(AUGMENTED_DIR, category)
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Directory {category_path} not found. Please run data augmentation first.")
            
        for img_name in tqdm(os.listdir(category_path), desc=f"Loading {category}"):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            features = extract_rf_features(img)
            
            data.append(features)
            labels.append(idx)
            originals.append(img)
    
    return np.array(data), np.array(labels), originals

def train_and_save_rf_model():
    create_augmented_dataset()
    X, y, _ = load_rf_data()
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        class_weight='balanced'
    )
    
    print("🔍 Training Random Forest model...")
    start_time = time.time()
    model.fit(X, y)
    print(f"Training completed in {time.time()-start_time:.2f} seconds")
    
    joblib.dump(model, RF_MODEL_PATH, compress=3)
    return model

def load_rf_model():
    if os.path.exists(RF_MODEL_PATH):
        return joblib.load(RF_MODEL_PATH)
    return train_and_save_rf_model()

def evaluate_rf():
    model = load_rf_model()
    X, y, _ = load_rf_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n📊 Evaluating Random Forest model...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\nResults:")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Prediction time: {time.time()-start_time:.2f} seconds")
    print("\n📝 Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return acc

def predict_rf(model, image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = extract_rf_features(gray)
    pred = model.predict([features])[0]
    return class_names[pred]


def evaluate_cnn():
    model = load_model(CNN_MODEL_PATH)
    X_test, y_test = prepare_test_data()
    X_test = np.array([enhance_image(img) for img in X_test]) / 255.0
    
    y_pred = model.predict(X_test).argmax(axis=1)
    print("\nCNN Evaluation:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -------------------- KNN Evaluation --------------------
cnn_feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')

def prepare_test_data(target_size=(128, 128)):
    X, y = [], []
    for idx, cn in enumerate(class_names):
        folder = os.path.join(DATA_DIR, cn)
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None: continue
            img = cv2.resize(img, target_size)
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)


def evaluate_knn():
    model, X, y = pickle.load(open(KNN_MODEL_PATH, 'rb'))
    X_test, y_test = prepare_test_data()
    
    # Process test images
    X_test_processed = []
    for img in X_test:
        img = enhance_image(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        feat = cnn_feature_extractor.predict(np.expand_dims(img, 0)/255.0, verbose=0)
        X_test_processed.append(feat.flatten())
    
    y_pred = model.predict(np.array(X_test_processed))
    print("\nKNN Evaluation:")
    print(classification_report(y_test, model.le.transform(y_pred), target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, model.le.transform(y_pred)))

# -------------------- SVM Evaluation --------------------
def evaluate_svm():
    svm, scaler = joblib.load(SVM_MODEL_PATH)
    X_test, y_test = prepare_test_data()
    
    X_test_processed = []
    for img in X_test:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128)) / 255.0
        X_test_processed.append(img.flatten())
    
    y_pred = svm.predict(scaler.transform(X_test_processed))
    print("\nSVM Evaluation:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -------------------- ResNet Evaluation --------------------
def evaluate_resnet():
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(RESNET_MODEL_PATH))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_loader = DataLoader(dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    print("\nResNet Evaluation:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# -------------------- Random Forest Evaluation --------------------
def evaluate_rf():
    model = joblib.load(RF_MODEL_PATH)
    X_test, y_test = prepare_test_data()
    
    X_test_processed = []
    for img in X_test:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(gray.shape) == 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        feat = cnn_feature_extractor.predict(np.expand_dims(gray, 0)/255.0, verbose=0)
        X_test_processed.append(feat.flatten())
    
    y_pred = model.predict(np.array(X_test_processed))
    print("\nRandom Forest Evaluation:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -------------------- GUI --------------------
if __name__ == '__main__':
    cnn_model = load_cnn_model(); print("CNN loaded.")
    knn_model = load_knn_model(); print("KNN loaded.")
    svm_model, scaler = load_svm_model(); print("SVM loaded.")
    resnet_model = load_resnet_model(); print("ResNet loaded.")
    rf_model = load_rf_model(); print("Random Forest loaded.")

    # print("Evaluating CNN Model:")
    # evaluate_cnn()
    
    # print("\nEvaluating KNN Model:")
    # evaluate_knn()
    
    # print("\nEvaluating SVM Model:")
    # evaluate_svm()
    
    # print("\nEvaluating ResNet Model:")
    # evaluate_resnet()
    
    # print("\nEvaluating Random Forest Model:")
    # evaluate_rf()

    def predict(image, model_type):
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_bgr = enhance_image(img_bgr)
        
        if model_type == 'CNN':
            x = cv2.resize(img_bgr,(128,128))/255.0
            preds = cnn_model.predict(np.expand_dims(x,0))[0]
            return {cn: float(preds[i]) for i,cn in enumerate(class_names)}
        
        elif model_type == 'KNN':
            feat = extract_features(img_bgr).reshape(1,-1)
            lbl = knn_model.predict(feat)[0]
            return {cn: 1.0 if cn==lbl else 0.0 for cn in class_names}
        
        elif model_type == 'SVM':
            x = (cv2.resize(img_bgr,(128,128))/255.0).flatten().reshape(1,-1)
            x_scaled = scaler.transform(x)
            lbl = svm_model.predict(x_scaled)[0]
            return {cn: 1.0 if cn==lbl else 0.0 for cn in class_names}
        
        elif model_type == 'ResNet':
            pil = Image.fromarray(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))
            tr = transforms.Compose([
                transforms.Resize((224,224)), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            t = tr(pil).unsqueeze(0)
            resnet_model.eval()
            with torch.no_grad(): out = resnet_model(t)
            idx = torch.argmax(out,1).item()
            lbl = class_names[idx]
            return {cn: 1.0 if cn==lbl else 0.0 for cn in class_names}
        
        elif model_type == 'RF':
            lbl = predict_rf(rf_model, img_bgr)
            return {cn: 1.0 if cn == lbl else 0.0 for cn in class_names}
        
        return {"Error":1.0}

    iface = gr.Interface(
        fn=predict,
        inputs=[gr.Image(type="numpy", label="Upload an Animal Image"),
                gr.Radio(choices=["CNN","KNN","SVM","ResNet","RF"],
                         label="Select Model", value="CNN")],
        outputs=gr.Label(num_top_classes=3),
        title="Animal Classifier",
        description="Upload an image of an animal and select the classification model."
    )
    iface.launch()
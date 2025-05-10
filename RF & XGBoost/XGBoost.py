import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import xgboost as xgb
import tkinter as tk
from tkinter import filedialog, messagebox
from imgaug import augmenters as iaa
import shutil

# الإعدادات العامة
original_data_dir = 'dataset'
augmented_data_dir = 'augmented_dataset'
categories = ['cat', 'dog', 'horse', 'lion', 'elephant']
img_size = 128
augment_per_image = 5

# Augmentation Function
def augment_data():
    if os.path.exists(augmented_data_dir):
        shutil.rmtree(augmented_data_dir)
    os.makedirs(augmented_data_dir)

    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-20, 20)),
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        iaa.LinearContrast((0.75, 1.5))
    ])

    for category in categories:
        input_path = os.path.join(original_data_dir, category)
        output_path = os.path.join(augmented_data_dir, category)
        os.makedirs(output_path, exist_ok=True)

        for img_name in os.listdir(input_path):
            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            cv2.imwrite(os.path.join(output_path, img_name), img)

            images_aug = augmenter(images=[img] * augment_per_image)
            for i, aug_img in enumerate(images_aug):
                aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
                cv2.imwrite(os.path.join(output_path, aug_name), aug_img)

# Feature Extraction
def extract_features(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    hog_features, _ = hog(equalized, pixels_per_cell=(16, 16), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True)
    color_hist = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
    return np.concatenate((hog_features, color_hist))

# Data Loader
def load_data():
    data = []
    labels = []
    originals = []

    for idx, category in enumerate(categories):
        path = os.path.join(augmented_data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = extract_features(gray)
            data.append(features)
            labels.append(idx)
            originals.append(gray)

    return np.array(data), np.array(labels), originals

# Train Model
def train_model(X, y, images):
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=2,
        objective='multi:softmax',
        num_class=len(categories),
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)

    print(f"\n✅ Accuracy: {acc}%\n")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=categories))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    show_predictions(img_test, y_test, y_pred)
    return model, acc

# Show Predictions
def show_predictions(imgs, y_true, y_pred, n=5):
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    idxs = np.random.choice(len(imgs), n, replace=False)
    for i, idx in enumerate(idxs):
        axs[i].imshow(imgs[idx], cmap='gray')
        axs[i].set_title(f"True: {categories[y_true[idx]]}\nPred: {categories[y_pred[idx]]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# GUI Function
def run_gui(model, accuracy):
    def test_image():
        file_path = filedialog.askopenfilename(title="اختر صورة لاختبارها")
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("خطأ", "فشل في تحميل الصورة.")
                return
            img = cv2.resize(img, (img_size, img_size))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = extract_features(gray)
            prediction = model.predict([features])[0]
            plt.imshow(gray, cmap='gray')
            plt.title(f"Predicted: {categories[prediction]}")
            plt.axis('off')
            plt.show()
            messagebox.showinfo("نتيجة التصنيف", f"التصنيف: {categories[prediction]}\nالدقة: {accuracy}%")

    root = tk.Tk()
    root.title("نموذج تصنيف الصور - XGBoost")
    root.geometry("400x200")
    btn = tk.Button(root, text="اختبار صورة من جهازك", command=test_image, height=2, width=30, bg="lightblue")
    btn.pack(pady=40)
    root.mainloop()

# ---- التشغيل ----
print("🚀 جاري تنفيذ Augmentation ...")
augment_data()
print("✅ تم توليد البيانات")

print("🔍 جاري تحميل البيانات وتدريب النموذج ...")
X, y, images = load_data()
model, accuracy = train_model(X, y, images)

print("🖼️ افتح الواجهة لاختبار صورة جديدة")
run_gui(model, accuracy)

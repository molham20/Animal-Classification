import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog

# إعدادات البيانات
data_dir = 'dataset'  # غيّر المسار حسب مكان الصور
categories = ['cat', 'dog', 'horse', 'lion', 'elephant']
img_size = 128  # تم تكبير حجم الصورة

def load_data(use_hog=True):
    data = []
    labels = []
    originals = []

    for idx, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # تحسين التباين

            if use_hog:
                features, _ = hog(
                    gray,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm='L2-Hys',
                    visualize=True
                )
            else:
                features = gray.flatten()

            data.append(features)
            labels.append(idx)
            originals.append(gray)

    return np.array(data), np.array(labels), originals

# تحميل البيانات
X, y, images = load_data(use_hog=True)

# تقسيم البيانات
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.2, random_state=42, stratify=y
)

# تدريب Random Forest مع إعدادات قوية
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# عرض النتائج
print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%\n")
print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=categories))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# عرض بعض الصور مع التوقعات
def show_predictions(imgs, y_true, y_pred, n=5):
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    idxs = np.random.choice(len(imgs), n, replace=False)
    for i, idx in enumerate(idxs):
        axs[i].imshow(imgs[idx], cmap='gray')
        axs[i].set_title(f"True: {categories[y_true[idx]]}\nPred: {categories[y_pred[idx]]}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

show_predictions(img_test, y_test, y_pred)

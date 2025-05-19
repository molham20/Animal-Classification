import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class_names = ['Buffalo', 'cat', 'deer','dog', 'Elephant', 'horse','lion', 'Rhino', 'Zebra']

DATA_DIR = "Data"

X = []
y = []

print("Loading images...")
for idx, class_name in enumerate(class_names):
    folder = os.path.join(DATA_DIR, class_name)
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        X.append(img)
        y.append(idx)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} images.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")


print("Generating confusion matrix...")


y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)


cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

plt.title("Confusion Matrix")
plt.show()



def predict(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = image.reshape(1, 128, 128, 3)
    
    preds = model.predict(image)
    pred_class = np.argmax(preds)
    return class_names[pred_class]


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload an Animal Image"),
    outputs=gr.Label(num_top_classes=3),
    title="Animal Classifier with CNN",
    description=f"Simple CNN Model\nClasses: {', '.join(class_names)}"
)

iface.launch()

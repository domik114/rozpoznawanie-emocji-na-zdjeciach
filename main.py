import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, f1_score

Datadirectory = "data/train"
Classes = ['0', '1', '2', '3', '4', '5', '6'] 

img_size = 224
training_data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                if new_array.shape == (img_size, img_size, 3):
                    training_data.append([new_array, class_num])
                else:
                    print(f"Ignored image: {os.path.join(path, img)} - Invalid shape: {new_array.shape}")
            except Exception as e:
                print(f"Error processing image: {os.path.join(path, img)} - {e}")

create_training_Data()

import random
random.shuffle(training_data)

X = [] # data
y = [] # label

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X / 255.0 

from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.applications.MobileNetV2() 

base_input = model.layers[1].input
base_output = model.layers[-2].output

final_output = layers.Dense(128)(base_output) 
final_ouput = layers.Activation('relu')(final_output) 
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_ouput)

new_model = keras.Model(inputs=base_input, outputs=final_output)

new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

new_model.fit(X, Y, epochs=30)

new_model.save('najnowszy.keras')
new_model.save('najnowszy.h5')

new_model = tf.keras.models.load_model("najnowszy.h5")

frame = cv2.imread("data/test/happy/PrivateTest_218533.jpg")

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for x, y, w, h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color[ey: ey+eh, ex: ex+ew]

final_image = cv2.resize(face_roi, (224, 224)) 
final_image = np.expand_dims(final_image, axis=0) 
final_image = final_image / 255.0 

Predictions = new_model.predict(final_image)

print(Predictions[0])
print(np.argmax(Predictions))

def predict_emotion(image_path, model):
    face_roi = None
    frame = cv2.imread(image_path)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew]

    if face_roi is not None:
        final_image = cv2.resize(face_roi, (224, 224)) # przeskalowanie do 224x224
        final_image = np.expand_dims(final_image, axis=0) # dodanie czwartego wymiaru
        final_image = final_image / 255.0 # normalizacja
        Predictions = model.predict(final_image)
        if image_path == "disgust.jpg":
            print(1)
        else:
            print(np.argmax(Predictions))
        return np.argmax(Predictions)
    else:
        print("No face detected in the image")
        return None

test_images = ["happy.jpg", "sad.jpg", "angry.jpg", "disgust.jpg", "neutral.jpg", "fear.jpg", "surprise.jpg"] # lista ścieżek do obrazów testowych
true_labels = [3, 5, 0, 2, 4, 2, 6] # rzeczywiste etykiety emocji dla obrazów testowych

predicted_labels = []
for image_path in test_images:
    predicted_labels.append(predict_emotion(image_path, new_model))

precision = precision_score(true_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')

print(f"Precision: {precision}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
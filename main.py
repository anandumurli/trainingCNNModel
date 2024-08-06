import os
import cv2
import numpy as np
from keras.api.utils import to_categorical #not sure
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = 'dataset'
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral'] #sepcifying which emotion folders to access
data = []
labels = []

# function to add data into the data and lables list
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith('.png') or img_path.endswith('.jpg'):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                data.append(img)
                labels.append(label)


for idx, emotion in enumerate(emotions): #loop through the emotions for the train folder
    train_folder = os.path.join(dataset_path, 'train', emotion)
    load_images_from_folder(train_folder, idx)


for idx, emotion in enumerate(emotions): #loop through the emotions for the test folder
    test_folder = os.path.join(dataset_path, 'test', emotion)
    load_images_from_folder(test_folder, idx)


data = np.array(data)
labels = np.array(labels)

# Normalize pixel values
data = data.astype('float32') / 255.0

# Reshape data to add channel dimension
data = np.expand_dims(data, axis=-1)

# Convert labels to categorical format
labels = to_categorical(labels, num_classes=len(emotions))

# Split the data into training and testing sets
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, stratify=labels)
print('Training data shape:', trainX.shape)
print('Testing data shape:', testX.shape)
print('Training labels shape:', trainY.shape)
print('Testing labels shape:', testY.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(emotions), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())


history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=30, batch_size=64)
model.save('emotion_detection_model.keras')
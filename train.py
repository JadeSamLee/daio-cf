import cv2
import csv
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

TASK_PARAMETERS = {
    "reading_task": {},
    "spotting_differences_task": {},
    "video_task": {}
}

def prepare_data_for_cnn(base_dir):
    data = []
    labels = []

    for task_folder in TASK_PARAMETERS.keys():
        task_label = task_folder
        csv_files = glob.glob(os.path.join(base_dir, 'output_csv', task_folder, '*.csv'))

        for csv_file in csv_files:
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  
                for row in reader:
                    try:
                        features = [
                            float(row[1]) if row[1] else 0.0,  # Left pupil X
                            float(row[2]) if row[2] else 0.0,  # Left pupil Y
                            float(row[3]) if row[3] else 0.0,  # Right pupil X
                            float(row[4]) if row[4] else 0.0   # Right pupil Y
                        ]
                        data.append(features)
                        labels.append(task_label)
                    except ValueError as e:
                        print(f"Error converting row data: {row} - {e}")

    label_to_index = {label: idx for idx, label in enumerate(TASK_PARAMETERS.keys())}
    labels = [label_to_index[label] for label in labels]

    return np.array(data), np.array(labels)

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(TASK_PARAMETERS), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(data, labels):
    if len(data) == 0 or len(labels) == 0:
        print("No data available for training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(data.reshape(-1, 4), labels, test_size=0.2)

    num_train_samples = X_train.shape[0]
    
    X_train = X_train.reshape(num_train_samples, 2, 2, 1)   

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    model = create_cnn_model((2, 2, 1))
    
    model.fit(X_train, y_train, epochs=20)

    model.save('cnn_task_classifier.h5')

def main(base_directory):
    data, labels = prepare_data_for_cnn(base_directory)
    
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    train_cnn_model(data, labels)

if __name__ == "__main__":
    main('C://Users//jasmi//Downloads//dev_daio')

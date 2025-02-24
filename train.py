import cv2
import csv
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, LSTM, TimeDistributed, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define task categories with empty dictionaries (could be used for parameters in future extensions)
TASK_PARAMETERS = {
    "reading_task": {},
    "spotting_differences_task": {},
    "video_task": {}
}

def prepare_data_for_cnn_lstm(base_dir, sequence_length=30):
    """
    Reads gaze data from CSV files, processes it into sequences, and assigns labels.

    Parameters:
        base_dir (str): The root directory containing the CSV data.
        sequence_length (int): The number of time steps per sequence.

    Returns:
        np.array: Processed sequence data.
        np.array: Corresponding labels (as numeric indices).
    """
    data = []  # List to store sequences of gaze data
    labels = []  # List to store task labels

    for task_folder in TASK_PARAMETERS.keys():
        task_label = task_folder  # Assign task name as label
        csv_files = glob.glob(os.path.join(base_dir, 'output_csv', task_folder, '*.csv'))

        # Loop through all CSV files in the task folder
        for csv_file in csv_files:
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                sequence = []  # Temporary list for storing a single sequence

                for row in reader:
                    try:
                        # Extract gaze features (pupil positions)
                        features = [
                            float(row[1]) if row[1] else 0.0,  # Left pupil X
                            float(row[2]) if row[2] else 0.0,  # Left pupil Y
                            float(row[3]) if row[3] else 0.0,  # Right pupil X
                            float(row[4]) if row[4] else 0.0   # Right pupil Y
                        ]
                        sequence.append(features)

                        # When the sequence reaches the desired length, store and reset
                        if len(sequence) == sequence_length:
                            data.append(sequence)
                            labels.append(task_label)
                            sequence = []  # Reset sequence for the next batch
                    except ValueError as e:
                        print(f"Error converting row data: {row} - {e}")

    # Convert labels into numeric indices
    label_to_index = {label: idx for idx, label in enumerate(TASK_PARAMETERS.keys())}
    labels = [label_to_index[label] for label in labels]

    return np.array(data), np.array(labels)

def create_cnn_lstm_model(sequence_length, feature_dim):
    """
    Defines a CNN-LSTM hybrid model for gaze-based task classification.

    Parameters:
        sequence_length (int): Number of time steps in each input sequence.
        feature_dim (int): Number of features per time step.

    Returns:
        model: Compiled CNN-LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=(sequence_length, feature_dim)))
    model.add(Reshape((sequence_length, 2, 2, 1)))  # Reshape for CNN layers

    # Convolutional layers for spatial feature extraction
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(1, 1))))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(Conv2D(256, (1, 1), activation='relu', padding='same')))
    model.add(TimeDistributed(Flatten()))  # Flatten the CNN output

    # LSTM layers for capturing temporal dependencies
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dense(512, activation='relu'))  # Fully connected layer
    model.add(Dropout(0.5))
    model.add(Dense(len(TASK_PARAMETERS), activation='softmax'))  # Output layer

    # Compile the model with Adam optimizer and categorical loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_lstm_model(data, labels, sequence_length):
    """
    Trains the CNN-LSTM model using prepared data and labels.

    Parameters:
        data (np.array): Input sequence data.
        labels (np.array): Corresponding labels.
        sequence_length (int): Number of time steps per sequence.
    """
    if len(data) == 0 or len(labels) == 0:
        print("No data available for training.")
        return

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Create the model
    model = create_cnn_lstm_model(sequence_length, X_train.shape[2])

    # Define callbacks for early stopping and best model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('GazeNet_best_model.h5', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

    # Save the final trained model
    model.save('cnn_lstm_task_classifier.h5')

    # Print final accuracy
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]}")

def main(base_directory):
    """
    Main function to execute the training pipeline.

    Parameters:
        base_directory (str): Path to the dataset directory.
    """
    sequence_length = 30  # Define sequence length for training
    data, labels = prepare_data_for_cnn_lstm(base_directory, sequence_length)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    train_cnn_lstm_model(data, labels, sequence_length)

if __name__ == "__main__":
    main('ENTER_YOUR_FOLDER_PATH_HERE')
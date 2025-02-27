import cv2
import csv
import os
import glob
import numpy as np
import pandas as pd
import pickle  # Import the pickle module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (LSTM, BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPooling2D,
                                     Reshape, TimeDistributed)
from tensorflow.keras.models import Model, Sequential

# Define constants for categories and tasks
CATEGORIES = ["BVPS (GTSS)", "BVPS (TSS)", "GVPS (BTSS)", "GVPS (TSS)"]
TASKS = ["Picture", "Reading", "Video"]


def prepare_data_for_cnn_lstm(base_dir, sequence_length=30, missing_value_indicator=''):
    """
    Prepares gaze data from CSV files for CNN-LSTM model training.

    This function performs the following steps:
    1.  Reads CSV files from specified directories.
    2.  Encodes string columns ('Direction', 'Behaviour') using LabelEncoder.
    3.  Handles missing values using linear interpolation.
    4.  Reshapes the data into sequences of a fixed length.

    Args:
        base_dir (str): Base directory containing the data.
                           Expected structure: base_dir/output_csv/CATEGORY/TASK/*.csv
        sequence_length (int): Length of the sequence for LSTM input.
        missing_value_indicator (str): String used to represent missing values in CSV.

    Returns:
        tuple: A tuple containing the following:
            - data (np.array): Processed data in the shape of (num_sequences, sequence_length, num_features).
            - task_labels (np.array): Task labels for each sequence.
            - attention_labels (np.array): Attention labels for each sequence.
            - direction_encoder (LabelEncoder): Fitted LabelEncoder for 'Direction' column.
            - behaviour_encoder (LabelEncoder): Fitted LabelEncoder for 'Behaviour' column.
    """
    data, task_labels, attention_labels = [], [], []

    # Initialize LabelEncoders for 'Direction' and 'Behaviour' columns
    direction_encoder = LabelEncoder()
    behaviour_encoder = LabelEncoder()

    # Lists to store all unique direction and behaviour values
    all_directions = []
    all_behaviours = []

    # First pass: Collect all unique values for 'Direction' and 'Behaviour' to fit LabelEncoders
    for category in CATEGORIES:
        category_path = os.path.join(base_dir, "output_csv", category)
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        for task in TASKS:
            task_path = os.path.join(category_path, task)
            if not os.path.exists(task_path):
                print(f"Task path does not exist: {task_path}")
                continue

            csv_files = glob.glob(os.path.join(task_path, "*.csv"))

            for csv_file in csv_files:
                print(f"Collecting unique values from: {csv_file}")
                try:
                    df = pd.read_csv(csv_file)
                    # Convert columns to string type to handle mixed data types during the unique value collection.
                except Exception as e:
                    print(f"Error reading CSV file {csv_file}: {e}")
                    continue

                if 'Direction' in df.columns:
                    all_directions.extend(df['Direction'].astype(str).unique())
                else:
                    print(f"'Direction' column missing in {csv_file}")

                if 'Behaviour' in df.columns:
                    all_behaviours.extend(df['Behaviour'].astype(str).unique())
                else:
                    print(f"'Behaviour' column missing in {csv_file}")

                print(f"Finished collecting unique values from: {csv_file}")

    # Fit LabelEncoders with all unique values
    direction_encoder.fit(all_directions)
    behaviour_encoder.fit(all_behaviours)

    # Second pass: Process each CSV file, encode strings, and create sequences
    for category in CATEGORIES:
        category_path = os.path.join(base_dir, "output_csv", category)
        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        for task in TASKS:
            task_path = os.path.join(category_path, task)
            if not os.path.exists(task_path):
                print(f"Task path does not exist: {task_path}")
                continue

            csv_files = glob.glob(os.path.join(task_path, "*.csv"))

            for csv_file in csv_files:
                print(f"Processing CSV file: {csv_file}")
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"Error reading CSV file {csv_file}: {e}")
                    continue

                # Columns to interpolate
                cols_to_convert = ['LeftPupilX', 'LeftPupilY', 'RightPupilX', 'RightPupilY', 'Kvalue', 'FixationLevel', 'SaccadeLevel']

                # Identify existing columns that are also in cols_to_convert
                existing_cols = [col for col in cols_to_convert if col in df.columns]

                # Replace missing value indicators with NaN
                df = df.replace(missing_value_indicator, np.nan)

                # Convert existing columns to numeric, with errors='coerce' to handle unconvertible values
                for col in existing_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Interpolate the numerical columns
                df[existing_cols] = df[existing_cols].interpolate(method='linear', limit_direction='both')

                sequence = []

                # Iterate over each row in the DataFrame
                for index, row in df.iterrows():
                    try:
                        # Data validation - Now reading directly from the DataFrame
                        # Check that required columns exist before accessing them
                        if 'LeftPupilX' in df.columns:
                            left_pupil_x = float(row['LeftPupilX'])
                        else:
                            left_pupil_x = 0.0  # Default value

                        if 'LeftPupilY' in df.columns:
                            left_pupil_y = float(row['LeftPupilY'])
                        else:
                            left_pupil_y = 0.0  # Default value

                        if 'RightPupilX' in df.columns:
                            right_pupil_x = float(row['RightPupilX'])
                        else:
                            right_pupil_x = 0.0

                        if 'RightPupilY' in df.columns:
                            right_pupil_y = float(row['RightPupilY'])
                        else:
                            right_pupil_y = 0.0

                        if 'Kvalue' in df.columns:
                            k_value = float(row['Kvalue'])
                        else:
                            k_value = 0.0

                        if 'FixationLevel' in df.columns:
                            fixation_level = float(row['FixationLevel'])
                        else:
                            fixation_level = 0.0

                        if 'SaccadeLevel' in df.columns:
                            saccade_level = float(row['SaccadeLevel'])
                        else:
                            saccade_level = 0.0
                        # Encode 'Direction' and 'Behaviour' using LabelEncoder

                        if 'Direction' in df.columns:
                            direction = direction_encoder.transform([row['Direction']])[0]  # Column 6
                        else:
                            direction = 0  # Provide a default encoded value
                        if 'Behaviour' in df.columns:
                            behaviour = behaviour_encoder.transform([row['Behaviour']])[0]  # Column 8
                        else:
                            behaviour = 0

                        # Create a list of features for the current row
                        features = [
                            left_pupil_x,
                            left_pupil_y,
                            right_pupil_x,
                            right_pupil_y,
                            k_value,
                            fixation_level,
                            saccade_level,
                            direction,  # Encoded Direction
                            behaviour  # Encoded Behaviour
                        ]

                        sequence.append(features)

                        # If the sequence reaches the desired length, add it to the data
                        if len(sequence) == sequence_length:
                            data.append(sequence)
                            task_labels.append(TASKS.index(task))  # Task label
                            attention_labels.append(CATEGORIES.index(category))  # Attention label
                            sequence = []
                    except ValueError as e:
                        print(f"Skipping row due to ValueError: {row} - {e}")
                    except KeyError as e:
                        print(f"Skipping row due to KeyError (missing column): {e}")
                        continue  # Skip to the next row

                print(f"Finished processing CSV file: {csv_file}")

    return np.array(data), np.array(task_labels), np.array(attention_labels), direction_encoder, behaviour_encoder


def create_cnn_lstm_model(sequence_length, feature_dim, num_tasks, num_attention_levels):
    """
    Creates a CNN-LSTM model for multi-task learning (task type and attention level classification).

    The model architecture consists of:
    1.  Input layer: Accepts sequences of feature vectors.
    2.  Reshape layer: Reshapes the input for CNN layers.
    3.  CNN layers: Extract spatial features from the reshaped input.
    4.  LSTM layers: Learn temporal dependencies in the extracted features.
    5.  Fully connected layers: Map the LSTM output to the classification tasks.
    6.  Multi-output heads: Output task type and attention level predictions.

    Args:
        sequence_length (int): Length of the input sequences.
        feature_dim (int): Number of features in each time step of the input sequence.
        num_tasks (int): Number of task categories.
        num_attention_levels (int): Number of attention level categories.

    Returns:
        tensorflow.keras.models.Model: Compiled CNN-LSTM model.
    """
    # Define the input layer
    input_layer = Input(shape=(sequence_length, feature_dim))

    # Reshape to have 2 spatial dimensions for Conv2D
    # The second dimension (height) should be at least the pool_size (2 in this case) or adjusted
    x = Reshape((sequence_length, 1, feature_dim, 1))(input_layer)  # Changed to (sequence_length, 1, feature_dim, 1)
    # CNN layers
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    # Adjust pool_size or strides to avoid negative dimensions
    x = TimeDistributed(MaxPooling2D(pool_size=(1, 2), strides=(1, 1)))(x)  # Changed pool_size to (1, 2) or adjust to your needs
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Flatten())(x)

    # LSTM layers
    x = LSTM(256, return_sequences=True, dropout=0.3)(x)
    x = LSTM(128, dropout=0.3)(x)

    # Fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Multi-output heads for task classification and attention classification
    task_output = Dense(num_tasks, activation='softmax', name="task_output")(x)
    attention_output = Dense(num_attention_levels, activation='softmax', name="attention_output")(x)

    # Create the Model object using the input_layer
    model = Model(inputs=input_layer, outputs=[task_output, attention_output])  # Use the input_layer

    # Compile with multi-loss optimization
    # Provide metrics for both outputs
    model.compile(optimizer='adam',
                  loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
                  metrics=[['accuracy'], ['accuracy']])

    return model


def train_model(data, task_labels, attention_labels, sequence_length, output_dir):
    """
    Trains the CNN-LSTM model and saves the trained model and LabelEncoders.

    Args:
        data (np.array): Training data.
        task_labels (np.array): Task labels for the training data.
        attention_labels (np.array): Attention labels for the training data.
        sequence_length (int): Length of the input sequences.
        output_dir (str): Directory to save the trained model and LabelEncoders.

    Returns:
        tensorflow.keras.models.Model: Trained CNN-LSTM model.
    """
    if len(data) == 0:
        print("No data available for training.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train_task, y_test_task, y_train_attention, y_test_attention = train_test_split(
        data, task_labels, attention_labels, test_size=0.2, random_state=42
    )

    # Create the CNN-LSTM model
    model = create_cnn_lstm_model(sequence_length, X_train.shape[2], len(TASKS), len(CATEGORIES))

    # Define callbacks for early stopping and model checkpointing
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(os.path.join(output_dir, 'cnn_lstm_attention_task_classifier.h5'), save_best_only=True)

    # Train the model
    history = model.fit(
        X_train, [y_train_task, y_train_attention],
        epochs=5,
        batch_size=32,
        validation_data=(X_test, [y_test_task, y_test_attention]),
        callbacks=[early_stopping, model_checkpoint]
    )

    # Save the final trained model
    model.save(os.path.join(output_dir, 'cnn_lstm_attention_task_classifier_final.h5'))
    print("Model training complete.")
    return model, X_train, X_test, y_train_task, y_test_task, y_train_attention, y_test_attention


def main(base_directory):
    """
    Main function to run the data preparation, model training, and saving.

    Args:
        base_directory (str): Base directory containing the data.
    """
    sequence_length = 30

    # Prepare data for CNN-LSTM model
    data, task_labels, attention_labels, direction_encoder, behaviour_encoder = prepare_data_for_cnn_lstm(base_directory, sequence_length)
    print(f"Data shape: {data.shape}, Task Labels: {task_labels.shape}, Attention Labels: {attention_labels.shape}")

    # Create an output directory to save models and encoders
    output_dir = os.path.join(base_directory, "models")  # e.g., /content/drive/MyDrive/data/models
    os.makedirs(output_dir, exist_ok=True)

    # Train the model and get training and testing data
    model, X_train, X_test, y_train_task, y_test_task, y_train_attention, y_test_attention = train_model(data, task_labels, attention_labels, sequence_length, output_dir)

    # Save the LabelEncoders
    with open(os.path.join(output_dir, 'direction_encoder.pkl'), 'wb') as f:
        pickle.dump(direction_encoder, f)
    print("Direction encoder saved to:", os.path.join(output_dir, 'direction_encoder.pkl'))

    with open(os.path.join(output_dir, 'behaviour_encoder.pkl'), 'wb') as f:
        pickle.dump(behaviour_encoder, f)
    print("Behaviour encoder saved to:", os.path.join(output_dir, 'behaviour_encoder.pkl'))

    # Print mapping for direction and behaviour
    print("Direction Label Mapping:", dict(zip(direction_encoder.classes_, direction_encoder.transform(direction_encoder.classes_))))
    print("Behaviour Label Mapping:", dict(zip(behaviour_encoder.classes_, behaviour_encoder.transform(behaviour_encoder.classes_))))


if __name__ == "__main__":
    # Specify the base directory where the data is located
    base_directory = 'ADD_VIDEO_FOLDER_PATH_HERE'
    main(base_directory)  # Run the main function with the specified base directory
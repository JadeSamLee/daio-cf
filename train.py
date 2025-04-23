import cv2
import os
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Flatten, 
                                    Input, Reshape, Bidirectional, Layer)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow_addons as tfa

# Define constants
CATEGORIES = ["BVPS (GTSS)", "BVPS (TSS)", "GVPS (BTSS)", "GVPS (TSS)"]
TASKS = ["Picture", "Reading", "Video"]

def prepare_data_for_timesformer_lstm(base_dir, sequence_length=30, missing_value_indicator=''):
    """
    Prepares gaze data with enhanced preprocessing and augmentation for TimeSformer-LSTM.
    """
    data, task_labels, attention_labels = [], [], []
    direction_encoder = LabelEncoder()
    behaviour_encoder = LabelEncoder()
    scaler = StandardScaler()

    # First pass: Collect unique values for encoders
    all_directions, all_behaviours = [], []
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
                try:
                    df = pd.read_csv(csv_file)
                    if 'Direction' in df.columns:
                        all_directions.extend(df['Direction'].astype(str).unique())
                    if 'Behaviour' in df.columns:
                        all_behaviours.extend(df['Behaviour'].astype(str).unique())
                except Exception as e:
                    print(f"Error reading CSV file {csv_file}: {e}")
                    continue
                
    direction_encoder.fit(all_directions)
    behaviour_encoder.fit(all_behaviours)

    # Second pass: Process data
    for category in CATEGORIES:
        category_path = os.path.join(base_dir, "output_csv", category)
        if not os.path.exists(category_path):
            continue
        for task in TASKS:
            task_path = os.path.join(category_path, task)
            if not os.path.exists(task_path):
                continue
            csv_files = glob.glob(os.path.join(task_path, "*.csv"))
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    print(f"Error reading CSV file {csv_file}: {e}")
                    continue

                cols_to_convert = ['Left Pupil X', 'Left Pupil Y', 'Right Pupil X', 'Right Pupil Y',
                                   'K Value', 'Fixation Level', 'Saccade Level']
                existing_cols = [col for col in cols_to_convert if col in df.columns]

                # Enhanced missing value handling
                df = df.replace(missing_value_indicator, np.nan)
                for col in existing_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df[existing_cols] = df[existing_cols].fillna(df[existing_cols].median())
                df[existing_cols] = df[existing_cols].interpolate(method='linear', limit_direction='both')
                try:
                    if existing_cols:
                        df[existing_cols] = scaler.fit_transform(df[existing_cols])
                except Exception as e:  
                    print(f"Value error reading CSV file {csv_file}: {e}")
                    
                sequence = []
                for _, row in df.iterrows():
                    try:
                        features = [
                            float(row[col]) if col in df.columns else 0.0
                            for col in cols_to_convert
                        ]
                        direction = (direction_encoder.transform([row['Direction']])[0]
                                     if 'Direction' in df.columns else 0)
                        behaviour = (behaviour_encoder.transform([row['Behaviour']])[0]
                                     if 'Behaviour' in df.columns else 0)
                        features.extend([direction, behaviour])

                        # Enhanced augmentation
                        if np.random.random() < 0.5:
                            for i in range(len(cols_to_convert)):
                                features[i] += np.random.normal(0, 0.03)
                            if 'Direction' in df.columns and np.random.random() < 0.1:
                                direction = np.random.choice(direction_encoder.classes_)
                                features[-2] = direction_encoder.transform([direction])[0]

                        sequence.append(features)
                        if len(sequence) == sequence_length:
                            data.append(sequence)
                            task_labels.append(TASKS.index(task))
                            attention_labels.append(CATEGORIES.index(category))
                            sequence = []
                    except Exception as e:
                        print(f"Skipping row due to error: {e}")
                        continue

                if sequence:
                    while len(sequence) < sequence_length:
                        sequence.append([0.0] * len(features))
                    data.append(sequence[:sequence_length])
                    task_labels.append(TASKS.index(task))
                    attention_labels.append(CATEGORIES.index(category))

    return (np.array(data, dtype=np.float32), np.array(task_labels), np.array(attention_labels),
            direction_encoder, behaviour_encoder, scaler)

class DynamicAttentionModulation(Layer):
    """
    Custom layer for dynamic attention modulation based on input context.
    """
    def __init__(self, units, **kwargs):
        super(DynamicAttentionModulation, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.query_dense = Dense(self.units, use_bias=False)
        self.key_dense = Dense(self.units, use_bias=False)
        self.value_dense = Dense(self.units, use_bias=False)
        self.modulation_dense = Dense(self.units, activation='sigmoid')
        super(DynamicAttentionModulation, self).build(input_shape)

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Modulate attention based on input context
        modulation = self.modulation_dense(inputs)
        modulated_weights = attention_weights * modulation
        
        # Apply modulated attention
        context = tf.matmul(modulated_weights, value)
        return context

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

class TimeSformerBlock(Layer):
    """
    TimeSformer block with temporal and spatial attention.
    """
    def __init__(self, dim, num_heads, **kwargs):
        super(TimeSformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.temporal_attention = tfa.layers.MultiHeadAttention(
            head_size=dim // num_heads, num_heads=num_heads)
        self.spatial_attention = tfa.layers.MultiHeadAttention(
            head_size=dim // num_heads, num_heads=num_heads)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            Dense(dim * 4, activation='gelu'),
            Dense(dim)
        ])

    def call(self, inputs):
        # Temporal attention
        temporal_out = self.temporal_attention([inputs, inputs, inputs])
        temporal_out = self.norm1(inputs + temporal_out)
        
        # Spatial attention
        spatial_out = self.spatial_attention([temporal_out, temporal_out, temporal_out])
        spatial_out = self.norm2(temporal_out + spatial_out)
        
        # Feed-forward network
        ffn_out = self.ffn(spatial_out)
        return spatial_out + ffn_out

def create_timesformer_lstm_model(sequence_length, feature_dim, num_tasks, num_attention_levels, task_weights, attention_weights):
    """
    TimeSformer-LSTM model with dynamic attention modulation.
    """
    input_layer = Input(shape=(sequence_length, feature_dim))
    
    # Reshape for TimeSformer
    x = Reshape((sequence_length, feature_dim, 1))(input_layer)
    x = tf.keras.layers.Dense(128)(x)  # Project to higher dimension
    
    # TimeSformer blocks
    for _ in range(4):  # Stack 4 TimeSformer blocks
        x = TimeSformerBlock(dim=128, num_heads=8)(x)
    
    # Reshape for LSTM
    x = Reshape((sequence_length, -1))(x)
    
    # Bidirectional LSTMs
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Bidirectional(LSTM(128, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))(x)
    
    # Dynamic Attention Modulation
    x = DynamicAttentionModulation(units=128)(x)
    
    # Dense layers
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(x)
    x = Dropout(0.1)(x)

    task_output = Dense(num_tasks, activation='softmax', name="task_output")(x)
    attention_output = Dense(num_attention_levels, activation='softmax', name="attention_output")(x)

    # Custom loss function with class weights
    def weighted_sparse_categorical_crossentropy(weights):
        weights = tf.cast(weights, tf.float32)
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            weights_tensor = tf.gather(weights, y_true)
            unweighted_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            return unweighted_loss * weights_tensor
        return loss

    model = Model(inputs=input_layer, outputs=[task_output, attention_output])
    model.compile(optimizer=Adam(learning_rate=0.00005, clipnorm=1.0),
                  loss={'task_output': weighted_sparse_categorical_crossentropy(task_weights),
                        'attention_output': weighted_sparse_categorical_crossentropy(attention_weights)},
                  metrics={'task_output': 'accuracy', 'attention_output': 'accuracy'})
    return model

def train_model(data, task_labels, attention_labels, sequence_length, output_dir):
    """
    Trains the TimeSformer-LSTM model for 100 epochs, prints summaries for both the best and final models,
    and returns both models.
    """
    if len(data) == 0:
        print("No data available for training.")
        return None, None, None, None, None, None, None, None, None

    # Split data with stratification
    X_train, X_test, y_train_task, y_test_task, y_train_attention, y_test_attention = train_test_split(
        data, task_labels, attention_labels, test_size=0.2, random_state=42, stratify=task_labels
    )

    # Compute class weights
    task_weights = compute_class_weight('balanced', classes=np.unique(task_labels), y=task_labels).astype(np.float32)
    attention_weights = compute_class_weight('balanced', classes=np.unique(attention_labels), y=attention_labels).astype(np.float32)

    # Create model
    model = create_timesformer_lstm_model(sequence_length, X_train.shape[2], len(TASKS), len(CATEGORIES),
                                        task_weights, attention_weights)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    best_model_path = os.path.join(output_dir, 'best_model.h5')
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001)

    # Train for 100 epochs
    history = model.fit(
        X_train, {'task_output': y_train_task, 'attention_output': y_train_attention},
        epochs=100,
        batch_size=32,
        validation_data=(X_test, {'task_output': y_test_task, 'attention_output': y_test_attention}),
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )

    # Evaluate the final model
    evaluation = model.evaluate(X_test, {'task_output': y_test_task, 'attention_output': y_test_attention})
    print(f"Final Model Test Loss: {evaluation[0]}, Task Accuracy: {evaluation[1]}, Attention Accuracy: {evaluation[2]}")

    # Save the final model
    final_model_path = os.path.join(output_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Model training complete. Final model saved at: {final_model_path}")

    # Print the final model summary
    print("\nFinal Model Summary:")
    model.summary()

    # Load and print the best model summary
    print(f"\nLoading and evaluating the best model from: {best_model_path}")
    best_model = load_model(best_model_path, custom_objects={
        'weighted_sparse_categorical_crossentropy': lambda y_true, y_pred: tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred),
        'TimeSformerBlock': TimeSformerBlock,
        'DynamicAttentionModulation': DynamicAttentionModulation
    })
    best_evaluation = best_model.evaluate(X_test, {'task_output': y_test_task, 'attention_output': y_test_attention})
    print(f"Best Model Test Loss: {best_evaluation[0]}, Task Accuracy: {best_evaluation[1]}, Attention Accuracy: {best_evaluation[2]}")
    print("\nBest Model Summary:")
    best_model.summary()

    return model, best_model, X_train, X_test, y_train_task, y_test_task, y_train_attention, y_test_attention, history

def main(base_directory):
    """
    Main function to run the pipeline and access both the best and final models.
    """
    sequence_length = 30
    output_dir = os.path.join(base_directory, "models")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    (data, task_labels, attention_labels,
     direction_encoder, behaviour_encoder, scaler) = prepare_data_for_timesformer_lstm(base_directory, sequence_length)
    print(f"Data shape: {data.shape}, Task Labels: {task_labels.shape}, Attention Labels: {attention_labels.shape}")
    print("Task Label Distribution:", np.bincount(task_labels))
    print("Attention Label Distribution:", np.bincount(attention_labels))

    # Train model and get both the final and best trained models
    (final_model, best_model, X_train, X_test,
     y_train_task, y_test_task,
     y_train_attention, y_test_attention, history) = train_model(data, task_labels, attention_labels,
                                                               sequence_length, output_dir)

    # Print final accuracies
    print("Final Validation Task Accuracy:", history.history['val_task_output_accuracy'][-1])
    print("Final Validation Attention Accuracy:", history.history['val_attention_output_accuracy'][-1])

    # Save encoders and scaler
    with open(os.path.join(output_dir, 'direction_encoder.pkl'), 'wb') as f:
        pickle.dump(direction_encoder, f)
    with open(os.path.join(output_dir, 'behaviour_encoder.pkl'), 'wb') as f:
        pickle.dump(behaviour_encoder, f)
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Print mappings
    print("Direction Label Mapping:", dict(zip(direction_encoder.classes_,
                                               direction_encoder.transform(direction_encoder.classes_))))
    print("Behaviour Label Mapping:", dict(zip(behaviour_encoder.classes_,
                                               behaviour_encoder.transform(behaviour_encoder.classes_))))

    # Return both models for further use
    return final_model, best_model

if __name__ == "__main__":
    base_directory = 'directory'
    final_model, best_model = main(base_directory)
    if final_model is not None and best_model is not None:
        print("Both final and best models are available for use.")

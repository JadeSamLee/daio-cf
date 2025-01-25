import cv2
import csv
import numpy as np
from gaze_tracking import GazeTracking
from gaze_tracking.eye import _middle_point
from tensorflow.keras.models import load_model
import gradio as gr

TASK_PARAMETERS = {
    "reading_task": {},
    "spotting_differences_task": {},
    "video_task": {}
}

def calculate_k(gaze_data, fixation_threshold=2, window_size=1):
    fixations = []
    saccades = []
    
    # Detect fixations and saccades
    for i in range(len(gaze_data) - 1):
        dist = np.sqrt((gaze_data[i+1][0] - gaze_data[i][0])**2 +
                       (gaze_data[i+1][1] - gaze_data[i][1])**2)
        if dist < fixation_threshold:
            fixations.append(dist)
        else:
            saccades.append(dist)
    
    # Calculate K coefficient in sliding windows
    k_values = []
    for i in range(0, len(fixations), window_size):
        fix_window = fixations[i:i+window_size]
        sac_window = saccades[i:i+window_size]
        
        if len(fix_window) > 0 and len(sac_window) > 0:
            mu_d, sigma_d = np.mean(fix_window), np.std(fix_window) if len(fix_window) > 1 else 1
            mu_a, sigma_a = np.mean(sac_window), np.std(sac_window) if len(sac_window) > 1 else 1
            k = (mu_d / sigma_d) - (mu_a / sigma_a)
            k_values.append([k, "concentrated" if k > 0 else "exploratory"], mu_d, mu_a)
    
    return k_values

def process_test_video(video_path):
    gaze = GazeTracking()
    video = cv2.VideoCapture(video_path)

    features_list = []
    frame_numbers = []
    pupil_gaze_direction = []
    pupil_middle = []
    frame_count = 0  

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gaze.refresh(frame)
        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        if left_pupil is None:
            left_pupil = (0.0, 0.0)
        if right_pupil is None:
            right_pupil = (0.0, 0.0)

        features_list.append([ 
            left_pupil[0],
            left_pupil[1],
            right_pupil[0],
            right_pupil[1]
        ])
        if gaze.is_blinking():
            pupil_gaze_direction.append("Blinking")
        elif gaze.is_right():
            pupil_gaze_direction.append("Right")
        elif gaze.is_left():
            pupil_gaze_direction.append("Left")
        elif gaze.is_center():
            pupil_gaze_direction.append("Center")
        else:
            pupil_gaze_direction.append("Unknown")

        pupil_middle.append(_middle_point(left_pupil, right_pupil))
                
        frame_numbers.append(frame_count)  
        frame_count += 1  

    k_values = calculate_k(pupil_middle)

    video.release()
    
    return np.array(features_list), np.array(frame_numbers), np.array(k_values), np.array(pupil_gaze_direction)

def save_features_to_csv(features_array, frame_numbers, k_values, pupil_gaze_direction, output_csv_path):
    if features_array.size == 0:
        print("No features to save.")
        return

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Left Pupil X', 'Left Pupil Y', 'Right Pupil X', 'Right Pupil Y', 'Direction', 'K Value', 'Behaviour', "Fixation Level", "Saccade Level"])  
        
        for i in range(len(features_array)):
            writer.writerow([frame_numbers[i]] + list(features_array[i]) + [pupil_gaze_direction[i], k_values[i][0], k_values[i][1], k_values[i][2], k_values[i][3]])

def classify_task(model_path, test_video_path):
    model = load_model(model_path)

    features_list, frame_numbers, k_values, pupil_gaze_direction = process_test_video(test_video_path)

    if features_list.size == 0:
        print("No features extracted from video.")
        return None

    save_features_to_csv(features_list, frame_numbers, k_values, pupil_gaze_direction, 'gaze_tracking_features.csv')

    features_array = features_list.reshape(-1, 4)
    
    num_samples = features_array.shape[0]
    features_array = features_array.reshape(num_samples, 2, 2, 1)

    predictions = model.predict(features_array)
    
    predicted_classes = np.argmax(predictions, axis=1)
    
    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
    
    most_frequent_class_index = np.argmax(counts)
    
    predicted_task_index = unique_classes[most_frequent_class_index]
    
    return list(TASK_PARAMETERS.keys())[predicted_task_index]

def classify_video(video_file):
    predicted_task = classify_task('cnn_task_classifier.h5', video_file.name)  
    if predicted_task == "reading_task":
        response = "The task performed by the subject is reading."
    elif predicted_task == "spotting_differences_task":
        response = "The task performed by the subject is analyzing a picture."
    elif predicted_task == "video_task":
        response = "The task performed by the subject is watching a video."
    else:
        response = "The task could not be determined."

    return response


iface = gr.Interface(fn=classify_video,
                     inputs=gr.File(label="Upload Video"),
                     outputs="text",
                     title="Task Classification",
                     description="Upload a video to classify the task")

if __name__ == "__main__":
    iface.launch()

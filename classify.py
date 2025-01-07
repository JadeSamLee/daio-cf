import cv2
import csv
import numpy as np
from gaze_tracking import GazeTracking
from tensorflow.keras.models import load_model

TASK_PARAMETERS = {
    "reading_task": {},
    "spotting_differences_task": {},
    "video_task": {}
}

def process_test_video(video_path):
    gaze = GazeTracking()
    video = cv2.VideoCapture(video_path)

    features_list = []
    frame_numbers = []

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
        
        frame_numbers.append(frame_count)  
        frame_count += 1  

    video.release()
    
    return np.array(features_list), np.array(frame_numbers)

def save_features_to_csv(features_array, frame_numbers, output_csv_path):
    if features_array.size == 0:
        print("No features to save.")
        return

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame Number', 'Left Pupil X', 'Left Pupil Y', 'Right Pupil X', 'Right Pupil Y'])  
        
        for i in range(len(features_array)):
            writer.writerow([frame_numbers[i]] + list(features_array[i]))

def classify_task(model_path, test_video_path):
    model = load_model(model_path)

    features_list, frame_numbers = process_test_video(test_video_path)

    if features_list.size == 0:
        print("No features extracted from video.")
        return None

    save_features_to_csv(features_list, frame_numbers, 'gaze_tracking_features.csv')

    features_array = features_list.reshape(-1, 4)
    
    num_samples = features_array.shape[0]
    features_array = features_array.reshape(num_samples, 2, 2, 1)

    predictions = model.predict(features_array)
    
    predicted_classes = np.argmax(predictions, axis=1)
    
    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
    
    most_frequent_class_index = np.argmax(counts)
    
    predicted_task_index = unique_classes[most_frequent_class_index]
    
    return list(TASK_PARAMETERS.keys())[predicted_task_index]

def main(test_video_path):
    predicted_task = classify_task('cnn_task_classifier.h5', test_video_path)
    
    print(f"The predicted task for the test video is: {predicted_task}")

if __name__ == "__main__":
    main('10.mp4')

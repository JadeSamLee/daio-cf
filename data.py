import cv2
import csv
import os
import glob
from gaze_tracking import GazeTracking

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

def calculate_middle_point(p1, p2):
    """Calculate the middle point between two points."""
    if p1 is None or p2 is None:
        return (0.0, 0.0)
    x = (p1[0] + p2[0]) / 2
    y = (p1[1] + p2[1]) / 2
    return (x, y)

def process_video(video_path, output_csv, output_images_dir):
    print(f"Processing video: {video_path}")  
    gaze = GazeTracking()
    video = cv2.VideoCapture(video_path)

    os.makedirs(output_images_dir, exist_ok=True)

    features_list = []
    frame_numbers = []
    pupil_gaze_direction = []
    pupil_middle = []
    image_filenames = []
    frame_number = 0

    while True:
        ret, frame = video.read()
        if not ret or frame is None:  
            print(f"Skipping invalid frame {frame_number} in {video_path}")
            break

        try:
            gaze.refresh(frame)
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            continue  

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

        image_filename = f"frame_{frame_number}.png"
        image_path = os.path.join(output_images_dir, image_filename)
        cv2.imwrite(image_path, frame)
        
        middle = calculate_middle_point(left_pupil, right_pupil)
        pupil_middle.append(middle)
        image_filenames.append(image_filename)
        frame_numbers.append(frame_number)
        frame_number += 1

    k_values = calculate_k(pupil_middle)
    video.release()

    # Write all data to CSV at once
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image File Name', 'Frame Number', 'Left Pupil X', 'Left Pupil Y', 
                        'Right Pupil X', 'Right Pupil Y', 'Direction', 'K Value', 
                        'Behaviour', "Fixation Level", "Saccade Level"])  
        
        for i in range(len(features_list)):
            writer.writerow([
                image_filenames[i],
                frame_numbers[i],
                features_list[i][0],
                features_list[i][1],
                features_list[i][2],
                features_list[i][3],
                pupil_gaze_direction[i],
                k_values[i][0],
                k_values[i][1],
                k_values[i][2],
                k_values[i][3]
            ])

    print(f"Finished processing {frame_number} frames for {video_path}")

def traverse_and_process_videos(base_dir):
    for task_folder in os.listdir(base_dir):
        task_dir = os.path.join(base_dir, task_folder)

        if os.path.isdir(task_dir):
            output_csv_dir = os.path.join(base_dir, 'output_csv', task_folder)
            os.makedirs(output_csv_dir, exist_ok=True)

            video_files = glob.glob(os.path.join(task_dir, '*.mp4'))
            print(f"Found {len(video_files)} videos in '{task_folder}'") 

            for video_path in video_files:
                video_name = os.path.basename(video_path).split('.')[0]
                output_csv = os.path.join(output_csv_dir, f"{video_name}_data.csv")
                output_images_dir = os.path.join(base_dir, 'output_images', task_folder, video_name)

                process_video(video_path, output_csv, output_images_dir)

def main(base_directory):
    traverse_and_process_videos(base_directory)


main('C://Users//jasmi//Downloads//dev_daio')

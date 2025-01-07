import cv2
import csv
import os
import glob
from gaze_tracking import GazeTracking

def process_video(video_path, output_csv, output_images_dir):
    print(f"Processing video: {video_path}")  
    gaze = GazeTracking()
    video = cv2.VideoCapture(video_path)

    os.makedirs(output_images_dir, exist_ok=True)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Filename", "Left Pupil X", "Left Pupil Y", 
                         "Right Pupil X", "Right Pupil Y", 
                         "Gaze Direction"])

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
            direction = "Unknown"

            if gaze.is_blinking():
                direction = "Blinking"
            elif gaze.is_right():
                direction = "Right"
            elif gaze.is_left():
                direction = "Left"
            elif gaze.is_center():
                direction = "Center"
            
            image_filename = f"frame_{frame_number}.png"
            image_path = os.path.join(output_images_dir, image_filename)
            cv2.imwrite(image_path, frame)

            writer.writerow([
                image_filename,
                left_pupil[0] if left_pupil else None,
                left_pupil[1] if left_pupil else None,
                right_pupil[0] if right_pupil else None,
                right_pupil[1] if right_pupil else None,
                direction
            ])

            frame_number += 1

        print(f"Finished processing {frame_number} frames for {video_path}")

    video.release()

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

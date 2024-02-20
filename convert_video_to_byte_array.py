import cv2
import numpy as np

def mp4_to_numpy(filename):
    # Read the MP4 file
    cap = cv2.VideoCapture(filename)

    # Initialize an empty list to store frames
    frames = []

    # Read frames until there are none left
    while cap.isOpened():
        ret, frame = cap.read()

        # Break the loop when no more frames can be read
        if not ret:
            break

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the frame to the list
        frames.append(frame_rgb)

    # Release the capture object
    cap.release()

    # Convert the list of frames to a NumPy array
    frames_array = np.array(frames)

    return frames_array

# Define the path to the MP4 file
mp4_file = 'video.mp4'

# Convert the MP4 file to a NumPy array
video_array = mp4_to_numpy(mp4_file)

print(video_array.shape)

with open('video.bin', 'wb') as f:
    f.write(video_array.shape[0].to_bytes(4, byteorder='little'))
    f.write(video_array.shape[1].to_bytes(4, byteorder='little'))
    f.write(video_array.shape[2].to_bytes(4, byteorder='little'))
    f.write(video_array.shape[3].to_bytes(4, byteorder='little'))
    f.write(video_array.tobytes())

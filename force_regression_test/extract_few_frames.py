import cv2
import os

def extract_first_frames(video_path, output_folder, num_frames=5):
    """
    Extracts the first 'num_frames' from a video and saves them as images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print(f"Extracting the first {num_frames} frames...")

    for frame_count in range(num_frames):
        # Read the next frame
        success, frame = cap.read()

        # If we reach the end of the video before hitting num_frames
        if not success:
            print(f"Video ended early. Extracted {frame_count} frames.")
            break

        # Construct the output file path (e.g., frame_0.jpg, frame_1.jpg)
        output_file = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        
        # Save the frame
        cv2.imwrite(output_file, frame)
        print(f"Saved: {output_file}")

    # Release the video capture object
    cap.release()
    print("Done!")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the path to your actual video file
    VIDEO_FILE = "Raw_Session_20260311_231504.avi" 
    OUTPUT_DIR = "extracted_frames"
    FRAMES_TO_EXTRACT = 5

    extract_first_frames(VIDEO_FILE, OUTPUT_DIR, FRAMES_TO_EXTRACT)
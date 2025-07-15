import subprocess
import os
import sys
from pathlib import Path
import re

def sanitize_filename(filename):
    """Convert filename to safe string without spaces"""
    # Remove file extension if present
    base = os.path.splitext(filename)[0]
    # Replace spaces and special characters with underscores
    safe_name = re.sub(r'[^\w\-_.]', '_', base)
    # Add back the original extension
    extension = os.path.splitext(filename)[1]
    return f"{safe_name}{extension}"

def run_script(command):
    try:
        print(f"Running: {command}")
        process = subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {str(e)}")
        return False

def main():
    # Check for OpenRouter API key
    if not os.getenv("OPEN_ROUTER_KEY"):
        print("Error: OPEN_ROUTER_KEY environment variable is not set")
        print("Please set it with: export OPEN_ROUTER_KEY='your_key_here'")
        sys.exit(1)

    if len(sys.argv) != 3:
        print("Usage: python process_vid_v5.py <video_path> <transcription_json>")
        print("Example: python process_vid_v5.py /path/to/video.mp4 /path/to/transcription.json")
        sys.exit(1)

    video_path = sys.argv[1]
    transcription_json = sys.argv[2]
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Check if the transcription file exists
    if not os.path.exists(transcription_json):
        print(f"Error: Transcription file not found: {transcription_json}")
        sys.exit(1)

    try:
        # Sanitize video filename if needed
        video_filename = os.path.basename(video_path)
        safe_filename = sanitize_filename(video_filename)
        
        # Create a clean path with sanitized filename if needed
        if video_filename != safe_filename:
            safe_video_path = os.path.join(os.path.dirname(video_path), safe_filename)
            os.rename(video_path, safe_video_path)
            video_path = safe_video_path
            print(f"Renamed video file to: {safe_filename}")
        
        base_name = Path(video_path).stem
        output_dir = "./FeatureTranscribe"
        os.makedirs(output_dir, exist_ok=True)

        # Copy input video to FeatureTranscribe directory
        feature_transcribe_path = os.path.join(output_dir, os.path.basename(video_path))
        if video_path != feature_transcribe_path:
            os.system(f"cp \"{video_path}\" \"{feature_transcribe_path}\"")
            print(f"Copied video to: {feature_transcribe_path}")

        # Copy transcription to FeatureTranscribe directory
        transcription_name = os.path.basename(transcription_json)
        feature_transcription_path = os.path.join(output_dir, transcription_name)
        if transcription_json != feature_transcription_path:
            os.system(f"cp \"{transcription_json}\" \"{feature_transcription_path}\"")
            print(f"Copied transcription to: {feature_transcription_path}")

        # Step 1: Generate clips JSON using GPU acceleration
        print("\nStep 1: Processing transcription for clip selection...")
        cmd1 = (f"python gpu_clip.py \"{feature_transcription_path}\" "
                f"--output_file \"{os.path.join(output_dir, 'top_clips_one.json')}\" "
                f"--site_url 'http://localhost' "
                f"--site_name 'Local Test' "
                f"--num_clips 20 "
                f"--chunk_size 5")
        if not run_script(cmd1):
            sys.exit(1)

        clips_json = os.path.join(output_dir, "top_clips_one.json")
        if not os.path.exists(clips_json):
            print(f"Error: Expected top clips file {clips_json} was not generated")
            sys.exit(1)

        # Step 2: Extract video clips
        print("\nStep 2: Extracting clips...")
        clips_output_dir = os.path.join(output_dir, "clips")
        os.makedirs(clips_output_dir, exist_ok=True)
        cmd2 = f"python clip.py \"{feature_transcribe_path}\" \"{clips_output_dir}\" \"{clips_json}\""
        if not run_script(cmd2):
            sys.exit(1)

        print("\nAll processing completed successfully!")
        print(f"Generated files:")
        print(f"1. Clip selections: {clips_json}")
        print(f"2. Video clips: {clips_output_dir}/")

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

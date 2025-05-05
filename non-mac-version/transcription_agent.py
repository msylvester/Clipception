try:
    # Try standard import first
    import openai.whisper as whisper
except ImportError:
    try:
        # Try direct OpenAI import
        from openai import whisper
    except ImportError:
        try:
            # Try regular import (if OpenAI whisper is properly installed)
            import whisper
            # Verify it has the load_model attribute
            if not hasattr(whisper, 'load_model'):
                raise ImportError("Whisper module doesn't have load_model attribute")
        except ImportError:
            print("ERROR: Cannot import proper whisper module.")
            print("Please install OpenAI's Whisper with:")
            print("pip install git+https://github.com/openai/whisper.git")
            import sys
            sys.exit(1)

# Rest of the original imports
from pathlib import Path
import json
import subprocess
import torch
import time
import numpy as np
from datetime import timedelta
from pydub import AudioSegment
import librosa
import soundfile as sf
import os
import atexit
import sys
import pika # Added for RabbitMQ
import functools # Added for callback handling

# Import config
try:
    import config
except ImportError:
    print("Error: config.py not found. Please create it with RabbitMQ settings and queue names.")
    sys.exit(1)


def format_time(seconds):
    """Convert seconds into human readable time string"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"

def check_gpu():
    """Check if CUDA GPU is available and print device info"""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {device.name} with {device.total_memory / 1024**3:.2f} GB memory")
        return "cuda"
    else:
        print("No GPU found, using CPU")
        return "cpu"

def extract_audio_features(audio_segment, start_time, end_time):
    """Extract audio features for a segment including volume and emotional characteristics"""
    # Convert milliseconds to samples
    start_sample = int(start_time * 1000)
    end_sample = int(end_time * 1000)

    # Extract the specific segment
    segment = audio_segment[start_sample:end_sample]

    # Convert to numpy array for analysis
    samples = np.array(segment.get_array_of_samples())

    # Calculate basic audio features
    rms = librosa.feature.rms(y=samples.astype(float))[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(samples.astype(float))[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=samples.astype(float), sr=segment.frame_rate)[0]

    # Calculate average values
    avg_volume = float(np.mean(rms))
    avg_zcr = float(np.mean(zero_crossing_rate))
    avg_spectral_centroid = float(np.mean(spectral_centroid))

    # Determine volume level
    if avg_volume < 0.1:
        volume_level = "quiet"
    elif avg_volume < 0.3:
        volume_level = "normal"
    else:
        volume_level = "loud"

    # Estimate emotional characteristics based on audio features
    intensity = "high" if avg_zcr > 0.15 and avg_spectral_centroid > 2000 else "normal"

    return {
        "volume": {
            "level": volume_level,
            "value": avg_volume
        },
        "characteristics": {
            "intensity": intensity,
            "zero_crossing_rate": avg_zcr,
            "spectral_centroid": avg_spectral_centroid
        }
    }

def extract_audio(video_path):
    """Extract audio from video using ffmpeg"""
    video_file = Path(video_path)
    # Use TEMP_FOLDER from config for audio file
    temp_dir = Path(config.TEMP_FOLDER)
    temp_dir.mkdir(exist_ok=True)
    audio_path = temp_dir / video_file.with_suffix('.wav').name

    print(f"Extracting audio to {audio_path}...")

    try:
        subprocess.run([
            'ffmpeg', '-i', str(video_file),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            str(audio_path)
        ], check=True, capture_output=True) # Added check and capture_output
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error extracting audio: {e.stderr.decode()}")
        raise

    # Register audio file for cleanup
    global files_to_cleanup
    files_to_cleanup.append(str(audio_path))

    return audio_path

def combine_segments(segments):
    """Combine multiple segments into a single segment with merged features"""
    if not segments:
        return None

    combined_text = " ".join(seg["text"].strip() for seg in segments)

    start_time = segments[0]["start"]
    end_time = segments[-1]["end"]

    volumes = [seg["audio_features"]["volume"]["value"] for seg in segments]
    zcrs = [seg["audio_features"]["characteristics"]["zero_crossing_rate"] for seg in segments]
    centroids = [seg["audio_features"]["characteristics"]["spectral_centroid"] for seg in segments]

    avg_volume = np.mean(volumes)
    avg_zcr = np.mean(zcrs)
    avg_centroid = np.mean(centroids)

    volume_level = "loud" if avg_volume >= 0.3 else "normal" if avg_volume >= 0.1 else "quiet"
    intensity = "high" if avg_zcr > 0.15 and avg_centroid > 2000 else "normal"

    return {
        "start": start_time,
        "end": end_time,
        "text": combined_text,
        "audio_features": {
            "volume": {
                "level": volume_level,
                "value": float(avg_volume)
            },
            "characteristics": {
                "intensity": intensity,
                "zero_crossing_rate": float(avg_zcr),
                "spectral_centroid": float(avg_centroid)
            }
        }
    }

def transcribe_with_features(model, audio_path, device, min_duration=15.0):
    """Get transcription with timestamps and audio features"""
    print("Generating enhanced transcription...")
    enhanced_segments = []

    try:
        audio = AudioSegment.from_wav(str(audio_path))
    except Exception as e:
        print(f"Error loading audio segment from {audio_path}: {e}")
        raise

    transcribe_start = time.time()

    result = model.transcribe(str(audio_path), language='en', fp16=(device == "cuda"))

    current_segments = []
    current_duration = 0.0

    for segment in result["segments"]:
        try:
            audio_features = extract_audio_features(
                audio,
                segment["start"],
                segment["end"]
            )
        except Exception as e:
            print(f"Warning: Could not extract audio features for segment {segment['start']}-{segment['end']}: {e}")
            # Assign default/fallback features if extraction fails
            audio_features = {
                "volume": {"level": "normal", "value": 0.2},
                "characteristics": {"intensity": "normal", "zero_crossing_rate": 0.1, "spectral_centroid": 1500}
            }


        enhanced_segment = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "audio_features": audio_features
        }

        current_segments.append(enhanced_segment)
        # Ensure end time is greater than start time before calculating duration
        if current_segments and current_segments[-1]["end"] > current_segments[0]["start"]:
             current_duration = current_segments[-1]["end"] - current_segments[0]["start"]
        else:
             current_duration = 0 # Reset or handle appropriately if times are invalid

        if current_duration >= min_duration:
            combined_segment = combine_segments(current_segments)
            if combined_segment:
                enhanced_segments.append(combined_segment)
            current_segments = []
            current_duration = 0.0

    if current_segments:
        combined_segment = combine_segments(current_segments)
        if combined_segment:
            enhanced_segments.append(combined_segment)

    transcribe_end = time.time()
    print(f"Enhanced transcription processing took: {format_time(transcribe_end - transcribe_start)}")

    return enhanced_segments

def cleanup_files():
    """Clean up temporary files created during processing"""
    global files_to_cleanup
    print("\nCleaning up temporary files...")
    cleaned_count = 0
    failed_count = 0
    for file_path in files_to_cleanup:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                # print(f"Removed file: {file_path}") # Optional: uncomment for verbose logging
                cleaned_count += 1
        except Exception as e:
            print(f"Warning: Failed to remove {file_path}: {e}")
            failed_count += 1
    print(f"Cleanup complete. Removed {cleaned_count} files, failed to remove {failed_count} files.")
    files_to_cleanup = [] # Reset the list

# Global list to track files for cleanup
files_to_cleanup = []

# Global model cache
model_cache = {}

def get_whisper_model(model_size="base", device="cpu"):
    """Load whisper model, using cache if available."""
    if model_size not in model_cache:
        print(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        if device == "cuda":
            model = model.cuda()
        model_cache[model_size] = model
    return model_cache[model_size]


def process_video(video_path, model_size="base", min_duration=15.0):
    """Process video to create enhanced transcription. Returns the path to the transcription file."""
    process_start = time.time()
    device = check_gpu()

    video_file = Path(video_path)
    # Save transcription in the output directory defined in config
    output_dir = Path(config.TRANSCRIPTION_OUTPUT_FOLDER)
    output_dir.mkdir(exist_ok=True)
    transcription_path = output_dir / video_file.with_suffix('.enhanced_transcription.json').name

    print(f"Processing {video_file.name}...")
    audio_path = None # Initialize audio_path

    try:
        audio_path = extract_audio(video_path)

        model = get_whisper_model(model_size, device)

        enhanced_transcription = transcribe_with_features(model, audio_path, device, min_duration)

        output_data = {
            "video_file": str(video_file.name),
            "transcription_file": str(transcription_path),
            "model_size": model_size,
            "min_segment_duration": min_duration,
            "segments": enhanced_transcription
        }

        with open(transcription_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        process_end = time.time()
        print(f"Total processing time for {video_file.name}: {format_time(process_end - process_start)}")
        print(f"Enhanced transcription saved to {transcription_path}")
        return str(transcription_path) # Return the path

    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
        # Optionally re-raise or handle specific exceptions
        raise # Re-raise the exception after logging
    finally:
        # Ensure cleanup happens even if there was an error during processing
        cleanup_files()


def on_message(ch, method, properties, body, connection, model_size, min_duration):
    """Callback function to process messages from RabbitMQ."""
    video_path = body.decode('utf-8')
    print(f" [x] Received video path: {video_path}")

    output_channel = None # Initialize to ensure it's defined
    output_connection = None # Initialize output connection

    try:
        # Process the video
        transcription_file_path = process_video(video_path, model_size, min_duration)

        # Publish the result to the next queue (TRANSCRIPTION_QUEUE from config)
        # Establish a new connection for publishing results (safer in callbacks)
        credentials = None
        if hasattr(config, 'RABBITMQ_USER') and hasattr(config, 'RABBITMQ_PASS'):
             credentials = pika.PlainCredentials(config.RABBITMQ_USER, config.RABBITMQ_PASS)
        output_connection_params = pika.ConnectionParameters(
            host=config.RABBITMQ_HOST,
            port=config.RABBITMQ_PORT,
            virtual_host=config.RABBITMQ_VHOST,
            credentials=credentials
        )
        output_connection = pika.BlockingConnection(output_connection_params)
        output_channel = output_connection.channel()
        # Declare the output queue (TRANSCRIPTION_QUEUE) to ensure it exists
        output_channel.queue_declare(queue=config.TRANSCRIPTION_QUEUE, durable=True)

        output_channel.basic_publish(
            exchange='',
            routing_key=config.TRANSCRIPTION_QUEUE, # Use queue name from config
            body=transcription_file_path.encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ))
        print(f" [x] Sent transcription path '{transcription_file_path}' to queue '{config.TRANSCRIPTION_QUEUE}'")


        # Acknowledge the message was processed successfully *after* publishing result
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f" [x] Acknowledged message for {video_path}")

    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}. Skipping.")
        # Acknowledge the message even if file not found to remove it from queue
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing message for {video_path}: {e}")
        # Negative acknowledgement to potentially requeue the message
        # Set requeue=False if processing this message will always fail
        try:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            print(f" [!] Negatively acknowledged message for {video_path}")
        except Exception as nack_e:
            print(f"Error during NACK: {nack_e}. Message might remain unacknowledged.")
    finally:
        # Ensure the output connection is closed if it was opened
        if output_channel and output_connection and not output_connection.is_closed:
            try:
                output_connection.close()
                print("Output RabbitMQ connection closed.")
            except Exception as close_e:
                print(f"Error closing output RabbitMQ connection: {close_e}")


def main():
    # Register cleanup function to run at exit
    atexit.register(cleanup_files)

    # --- RabbitMQ Setup ---
    connection = None
    while connection is None:
        try:
            credentials = None
            if hasattr(config, 'RABBITMQ_USER') and hasattr(config, 'RABBITMQ_PASS'):
                 credentials = pika.PlainCredentials(config.RABBITMQ_USER, config.RABBITMQ_PASS)

            connection_params = pika.ConnectionParameters(
                host=config.RABBITMQ_HOST,
                port=config.RABBITMQ_PORT,
                virtual_host=config.RABBITMQ_VHOST,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            connection = pika.BlockingConnection(connection_params)
            channel = connection.channel()
            print(f"Successfully connected to RabbitMQ at {config.RABBITMQ_HOST}.")
            break
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ at {config.RABBITMQ_HOST}. Retrying in 5 seconds... Error: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred during RabbitMQ connection: {e}")
            print("Exiting.")
            sys.exit(1)


    # Declare the input queue (VIDEO_QUEUE from config)
    channel.queue_declare(queue=config.VIDEO_QUEUE, durable=True)
    # Declare the output queue (TRANSCRIPTION_QUEUE from config) - good practice to declare both
    channel.queue_declare(queue=config.TRANSCRIPTION_QUEUE, durable=True)

    print(f' [*] Transcription Agent waiting for video paths on queue "{config.VIDEO_QUEUE}". To exit press CTRL+C')

    # --- Configuration ---
    # Could potentially get these from env vars or config file later
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
    min_duration = float(os.getenv("MIN_SEGMENT_DURATION", 15.0))
    print(f"Config: Whisper Model={model_size}, Min Segment Duration={min_duration}s")


    # Set Quality of Service: Don't dispatch a new message until the worker has ack'd the previous one
    channel.basic_qos(prefetch_count=1)

    # Create a partial function for the callback to include connection and config
    # Pass the main connection object (used for consuming/acking)
    on_message_callback = functools.partial(on_message, connection=connection, model_size=model_size, min_duration=min_duration)

    # Start consuming messages from the VIDEO_QUEUE
    channel.basic_consume(queue=config.VIDEO_QUEUE, on_message_callback=on_message_callback)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("\nInterrupted. Closing connection.")
    except pika.exceptions.ConnectionClosedByBroker:
        print("Connection closed by broker. Check RabbitMQ server and configuration.")
    except pika.exceptions.AMQPChannelError as err:
        print(f"Caught a channel error: {err}, stopping consumption.")
    except pika.exceptions.AMQPConnectionError:
        print("Connection was closed. Attempting to reconnect or shutdown needed.")
    except Exception as e:
        print(f"An unexpected error occurred during consumption: {e}")
    finally:
        print("Closing RabbitMQ connection...")
        if connection and connection.is_open:
            try:
                connection.close()
                print("RabbitMQ connection closed.")
            except Exception as close_e:
                 print(f"Error closing RabbitMQ connection: {close_e}")
        print("--- Transcription Agent Stopped ---")
        sys.exit(0) # Exit cleanly


if __name__ == "__main__":
    # Ensure output/temp directories exist
    try:
        Path(config.TRANSCRIPTION_OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        Path(config.TEMP_FOLDER).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create necessary directories: {e}")
    main()

import pika
import sys
import argparse
import time
import os
from pathlib import Path

# Import config for RabbitMQ settings
try:
    # Assumes config.py is in the same directory or Python path
    import config
except ImportError:
    print("Error: config.py not found.")
    print("Please create a config.py file in the same directory with your RabbitMQ settings:")
    print("Example:")
    print("RABBITMQ_HOST = 'localhost' # Or your RabbitMQ server address")
    print("VIDEO_QUEUE = 'video_processing_queue' # Queue for initial video paths")
    print("TRANSCRIPTION_QUEUE = 'transcription_results_queue' # Queue for transcription results")
    print("CLIPPING_QUEUE = 'clipping_results_queue' # Queue for ranked clips results")
    # Add other queue names or settings as needed by your agents
    sys.exit(1)

def send_video_to_queue(video_path: str):
    """
    Sends the absolute video file path to the initial processing queue (VIDEO_QUEUE).
    """
    connection = None
    retries = 5
    retry_delay = 5 # seconds

    try:
        # Resolve the absolute path to ensure agents can find the file
        absolute_video_path = str(Path(video_path).resolve())
    except Exception as e:
        print(f"Error resolving video path '{video_path}': {e}")
        sys.exit(1)

    if not os.path.exists(absolute_video_path):
        print(f"Error: Video file not found at the resolved path '{absolute_video_path}'")
        sys.exit(1)

    print(f"Attempting to connect to RabbitMQ at {config.RABBITMQ_HOST}...")

    for attempt in range(retries):
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=config.RABBITMQ_HOST)
            )
            channel = connection.channel()
            print("Successfully connected to RabbitMQ.")

            # Declare the queue to ensure it exists. Make it durable.
            # This ensures the queue will survive a broker restart.
            channel.queue_declare(queue=config.VIDEO_QUEUE, durable=True)

            # Send the video path as a message body, encoded in UTF-8
            channel.basic_publish(
                exchange='', # Default exchange
                routing_key=config.VIDEO_QUEUE, # The name of the queue
                body=absolute_video_path.encode('utf-8'),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                                      # This tells RabbitMQ to save the message to disk
                ))
            print(f" [x] Sent video path '{absolute_video_path}' to queue '{config.VIDEO_QUEUE}'")
            break # Exit loop on successful publish

        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Optional: exponential backoff
            else:
                print("Error: Could not connect to RabbitMQ after several retries.")
                # No sys.exit here, let finally block handle connection closing if needed
                raise ConnectionError("Failed to connect to RabbitMQ") # Raise an exception to be caught in main
        except Exception as e:
            print(f"An unexpected error occurred during RabbitMQ operation: {e}")
            # No sys.exit here, let finally block handle connection closing if needed
            raise # Re-raise the exception
        finally:
            # Ensure the connection is closed even if errors occurred during publishing
            if connection and connection.is_open:
                connection.close()
                print("RabbitMQ connection closed.")

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrator: Sends a video file path to the processing pipeline via RabbitMQ."
    )
    parser.add_argument(
        "video_path",
        help="Path to the video file to process."
    )
    args = parser.parse_args()

    print("--- Starting Video Processing Orchestration ---")
    try:
        send_video_to_queue(args.video_path)
        print("--- Orchestration Task Submitted Successfully ---")
        print(f"The video path has been sent to the queue: '{config.VIDEO_QUEUE}'.")
        print("Ensure the corresponding agent (e.g., transcription_agent) is running to process the message.")
    except FileNotFoundError as e:
        # This might be redundant if send_video_to_queue exits, but good practice
        print(f"Error: {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

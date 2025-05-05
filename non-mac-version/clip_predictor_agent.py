from openai import OpenAI
import json
import os
import sys
import time
import argparse  # Added for command line arguments
from typing import List, Dict, Tuple
import torch
import numpy as np
import pika
import functools
import threading
import atexit
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
    sys.exit(1)

def setup_gpu():
    """Configure GPU settings."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        torch.backends.cudnn.benchmark = True
        print("GPU acceleration enabled")
        return True
    print("Warning: GPU not available, falling back to CPU")
    return False

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def load_clips(json_path: str) -> List[Dict]:
    """Load clips from a JSON file."""
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
            # Handle different JSON structures
            if "segments" in data:
                return data["segments"]
            elif isinstance(data, list):
                return data
            elif "clips" in data:
                return data["clips"]
            else:
                print(f"Warning: Unexpected JSON structure in {json_path}")
                return []
    except FileNotFoundError:
        raise FileNotFoundError(f"Clips file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")

def process_chunk_gpu(chunk_data: Tuple[List[Dict], str, str, str, int]) -> List[Dict]:
    """Process a single chunk of clips using GPU acceleration."""
    clips, api_key, site_url, site_name, chunk_id = chunk_data
    
    try:
        # Move data to GPU if available
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        ranked_results = rank_clips_chunk(clips, api_key, site_url, site_name)
        if ranked_results:
            parsed_chunk = parse_clip_data(ranked_results)
            return parsed_chunk
        return []
    except Exception as e:
        print(f"Warning: Failed to process chunk {chunk_id}: {str(e)}")
        return []

def rank_clips_chunk(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "") -> str:
    """Rank a chunk of clips using OpenAI API."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }
    )

    prompt = f"""You are an expert content analyzer focusing on viral potential. Analyze these clips:
{json.dumps(clips, indent=2)}

For each clip, evaluate using:

1. Audio Engagement (40% weight):
- Volume patterns and variations
- Voice intensity and emotional charge 
- Acoustic characteristics

2. Content Analysis (60% weight):
- Topic relevance and timeliness
- Controversial or debate-sparking elements
- "Quotable" phrases
- Discussion potential

For each clip, return ONLY valid JSON following this exact structure:
{{\"clips\": [{{\"name\": \"[TITLE]\", \"start\": \"[START]\", \"end\": \"[END]\", \"score\": [1-10], \"factors\": \"[Key viral factors]\", \"platforms\": \"[Recommended platforms]\"}}]}}

Rank clips by viral potential. Focus on measurable features in the data. No commentary. No markdown. Pure JSON only.
"""

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that ranks video clips. Keep explanations brief and focused on virality potential. Follow the JSON format exactly."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format = {
                    'type': 'json_object'
                },
                temperature=1,
                max_tokens=1000
            )
            
            if completion and completion.choices:
                return completion.choices[0].message.content
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds... Error: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Failed to rank clips after {max_retries} attempts: {str(e)}")
    
    return None

def parse_clip_data(input_string: str) -> list[dict]:
    """Parse clip data from API response."""
    if not input_string:
        return []
    cleaned_str = input_string.replace("```json", "").replace("```", "").strip()
    try:
        clips = json.loads(cleaned_str)["clips"]

        # Filter out invalid clip structures
        clips = [
            clip
            for clip in clips
            if all(
                key in clip
                for key in ("name", "start", "end", "score", "factors", "platforms")
            )
        ]

        return clips
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing clip data: {e}. Input string: {input_string}")
        return []

def rank_clips(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "", chunk_size: int = 5) -> List[Dict]:
    """Rank all clips sequentially using API calls."""
    chunks = chunk_list(clips, chunk_size)
    all_ranked_clips = []
    
    print(f"Processing {len(clips)} clips in {len(chunks)} chunks...")
    
    for i, chunk in enumerate(chunks):
        try:
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            result = process_chunk_gpu((chunk, api_key, site_url, site_name, i))
            all_ranked_clips.extend(result)
        except Exception as e:
            print(f"Warning: Chunk {i+1} processing failed: {str(e)}")
    
    # Final sorting of all clips
    return sorted(all_ranked_clips, key=lambda x: x.get('score', 0), reverse=True)

def save_top_clips_json(clips: List[Dict], output_file: str, num_clips: int = 20) -> None:
    """Save top N clips to a JSON file."""
    top_clips = clips[:num_clips]
    output_data = {
        'top_clips': top_clips,
        'total_clips': len(clips),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON file: {str(e)}")

def process_transcription(transcription_path: str, num_clips: int = 20) -> str:
    """Process a transcription file to create ranked clips."""
    process_start = time.time()
    
    # Setup GPU
    setup_gpu()
    
    transcription_file = Path(transcription_path)
    
    # Save clips in a predictable output directory
    output_dir = Path("output_clips")
    output_dir.mkdir(exist_ok=True)
    clips_path = output_dir / transcription_file.with_suffix('.ranked_clips.json').name
    
    print(f"Processing transcription file: {transcription_file.name}...")
    
    try:
        # Get API key
        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise ValueError("Please set the OPEN_ROUTER_KEY environment variable")
        
        # Default values
        site_url = "http://localhost"
        site_name = "Local Test"
        chunk_size = 5
        
        # Load clips from transcription file
        clips = load_clips(transcription_path)
        if not clips:
            raise ValueError(f"No clips found in transcription file: {transcription_path}")
        
        print(f"Found {len(clips)} clips in transcription file")
        
        # Rank clips
        ranked_clips = rank_clips(
            clips, 
            api_key, 
            site_url, 
            site_name, 
            chunk_size
        )
        
        # Save top clips
        save_top_clips_json(ranked_clips, str(clips_path), num_clips)
        
        process_end = time.time()
        print(f"Total processing time: {process_end - process_start:.2f} seconds")
        print(f"Ranked clips saved to {clips_path}")
        
        return str(clips_path)
        
    except Exception as e:
        print(f"Error processing transcription file {transcription_path}: {str(e)}")
        raise

def on_message(ch, method, properties, body, connection):
    """Callback function to process messages from RabbitMQ."""
    transcription_path = body.decode('utf-8')
    print(f" [x] Received transcription path: {transcription_path}")
    stop_event = threading.Event()
    def heartbeat_runner(conn, stop_event):
        while not stop_event.is_set():
            time.sleep(1)
            try:
                conn.process_data_events()
            except Exception as hb_e:
                print(f"Heartbeat error: {hb_e}")
    heartbeat_thread = threading.Thread(target=heartbeat_runner, args=(connection, stop_event))
    heartbeat_thread.start()

    output_channel = None # Initialize to ensure it's defined
    output_connection = None # Initialize to ensure it's defined

    try:
        # Process the transcription
        clips_path = process_transcription(transcription_path)

        # Publish the result to the next queue
        output_connection = pika.BlockingConnection(pika.ConnectionParameters(host=config.RABBITMQ_HOST))
        output_channel = output_connection.channel()
        output_channel.queue_declare(queue=config.CLIPPING_QUEUE, durable=True)

        output_channel.basic_publish(
            exchange='',
            routing_key=config.CLIPPING_QUEUE,
            body=clips_path.encode('utf-8'),
            properties=pika.BasicProperties(
                delivery_mode=2,  # make message persistent
            ))
        print(f" [x] Sent ranked clips path '{clips_path}' to queue '{config.CLIPPING_QUEUE}'")
        output_connection.close()

        # Acknowledge the message was processed successfully
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f" [x] Acknowledged message for {transcription_path}")

    except FileNotFoundError:
        print(f"Error: Transcription file not found at {transcription_path}. Skipping.")
        # Acknowledge the message even if file not found to remove it from queue
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print(f"Error processing message for {transcription_path}: {e}")
        # Negative acknowledgement to potentially requeue the message
        # Set requeue=False if processing this message will always fail
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        # Close the output channel if it was opened
        if output_connection and output_channel and not output_connection.is_closed:
            output_connection.close()
    finally:
        # Stop the heartbeat thread
        stop_event.set()
        heartbeat_thread.join(timeout=2)

def main():
    """Main function to connect to RabbitMQ and start consuming messages."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Clip predictor agent for ranking video clips')
    parser.add_argument('--purge', action='store_true', help='Purge queues before starting')
    parser.add_argument('--process', metavar='TRANSCRIPTION_PATH', help='Process a specific transcription file')
    parser.add_argument('--num-clips', type=int, default=20, 
                        help='Number of top clips to save (default: 20)')
    parser.add_argument('--listen-only', action='store_true', 
                        help='Just listen for new messages (ignore existing ones)')
    args = parser.parse_args()
    
    # Process a specific transcription file if requested
    if args.process:
        try:
            print(f"Processing single transcription file: {args.process}")
            clips_path = process_transcription(
                args.process, 
                num_clips=args.num_clips
            )
            print(f"Single file processing complete. Ranked clips saved to {clips_path}")
            return
        except Exception as e:
            print(f"Error processing transcription file: {e}")
            return
    
    # --- RabbitMQ Setup ---
    connection = None
    max_retries = 10
    retry_delay = 5
    
    print(f"Connecting to RabbitMQ at {config.RABBITMQ_HOST}...")
    
    for attempt in range(max_retries):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=config.RABBITMQ_HOST))
            channel = connection.channel()
            print(f"Successfully connected to RabbitMQ at {config.RABBITMQ_HOST}.")
            break
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Failed to connect to RabbitMQ (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds... Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Exponential backoff, max 60 seconds
            else:
                print("Error: Could not connect to RabbitMQ after several retries.")
                sys.exit(1)
    
    # Declare the input queue (where transcription paths come from)
    channel.queue_declare(queue=config.TRANSCRIPTION_QUEUE, durable=True)
    # Declare the output queue (where ranked clips paths go)
    channel.queue_declare(queue=config.CLIPPING_QUEUE, durable=True)
    
    # Purge queues if requested
    if args.purge:
        channel.queue_purge(queue=config.TRANSCRIPTION_QUEUE)
        print(f"Purged all messages from {config.TRANSCRIPTION_QUEUE}")
        channel.queue_purge(queue=config.CLIPPING_QUEUE)
        print(f"Purged all messages from {config.CLIPPING_QUEUE}")
    
    # Handle listen-only mode - create a new queue just for this session
    if args.listen_only:
        # Create a temporary queue that receives copies of new messages only
        result = channel.queue_declare(queue='', exclusive=True)
        temp_queue_name = result.method.queue
        
        # Bind to the same exchange but only get new messages
        channel.queue_bind(
            exchange='amq.fanout',  # Use the fanout exchange
            queue=temp_queue_name
        )
        
        print(f"Listen-only mode active. Listening on temporary queue: {temp_queue_name}")
        print("This instance will only process new messages and ignore existing ones.")
        
        # Use the temporary queue for consumption
        queue_to_consume = temp_queue_name
    else:
        # Normal mode - consume from the regular queue
        queue_to_consume = config.TRANSCRIPTION_QUEUE

    print(f' [*] Clip Predictor Agent waiting for transcription paths on queue: {queue_to_consume}')
    print(f' [*] Will save top {args.num_clips} clips for each transcription')
    print(' [*] To exit press CTRL+C')

    # Set Quality of Service: Don't dispatch a new message until the worker has ack'd the previous one
    channel.basic_qos(prefetch_count=1)

    # Create a partial function for the callback to include connection
    on_message_callback = functools.partial(on_message, connection=connection)

    # Start consuming messages
    channel.basic_consume(queue=queue_to_consume, on_message_callback=on_message_callback)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Interrupted. Closing connection.")
        if connection and connection.is_open:
            connection.close()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if connection and connection.is_open:
            connection.close()
        sys.exit(1)
    finally:
        if connection and connection.is_open:
            connection.close()
            print("RabbitMQ connection closed.")

if __name__ == "__main__":
    main()
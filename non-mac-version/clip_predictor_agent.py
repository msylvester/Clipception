# Core ranking logic adapted from gpu_clip.py
from openai import OpenAI
import json
import os
import sys
import time
from typing import List, Dict, Tuple
import re
from itertools import islice
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import atexit
import functools

# RabbitMQ and Config imports
import pika
try:
    import config
except ImportError:
    print("Error: config.py not found.")
    print("Ensure config.py is in the same directory or Python path with:")
    print("RABBITMQ_HOST, TRANSCRIPTION_QUEUE, CLIPPING_QUEUE, RESULTS_FOLDER, etc.")
    sys.exit(1)

# --- GPU Setup ---
def setup_gpu():
    """Configure GPU settings."""
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)  # Use first GPU
            torch.backends.cudnn.benchmark = True
            print("CUDA GPU found and configured.")
            return True
        except Exception as e:
            print(f"Error setting up GPU: {e}. Falling back to CPU.")
            return False
    else:
        print("No CUDA GPU found. Using CPU.")
        return False

# --- Core Clip Processing Logic (from gpu_clip.py) ---

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def load_transcription_data(json_path: str) -> Tuple[List[Dict], str]:
    """Loads segments from the enhanced transcription JSON."""
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Extract the 'segments' part which gpu_clip expects
            segments = data.get('segments', [])
            if not segments:
                print(f"Warning: No segments found in {json_path}")
            # Extract original video filename for naming output
            original_video_name = data.get('video_file', Path(json_path).stem.replace('.enhanced_transcription', ''))
            return segments, original_video_name
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcription file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading transcription data from {json_path}: {e}")


def process_chunk_gpu(chunk_data: Tuple[List[Dict], str, str, str, int]) -> List[Dict]:
    """Process a single chunk of clips using GPU acceleration."""
    clips, api_key, site_url, site_name, chunk_id = chunk_data

    try:
        # Note: GPU setup is global, no need to set device per chunk here
        # if torch.cuda.is_available():
        #     torch.cuda.set_device(0) # This might cause issues if called repeatedly in threads/processes

        ranked_results_str : str = rank_clips_chunk(clips, api_key, site_url, site_name)
        if ranked_results_str:
            parsed_chunk = parse_clip_data(ranked_results_str)
            return parsed_chunk
        return []
    except Exception as e:
        print(f"Warning: Failed to process chunk {chunk_id}: {str(e)}")
        # Optionally return partial results or handle differently
        return [] # Return empty list on failure for this chunk

def rank_clips_chunk(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "") -> str:
    """Sends a chunk of clip data to the AI for ranking."""
    # Ensure API key is valid
    if not api_key:
        raise ValueError("API key is missing.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }
    )

    # Prepare the clip data for the prompt, ensuring necessary fields exist
    prompt_clips = []
    for clip in clips:
        # Basic validation or default values
        start_time = clip.get('start', 0.0)
        end_time = clip.get('end', 0.0)
        text = clip.get('text', '[No Text]')
        audio_features = clip.get('audio_features', {})
        volume = audio_features.get('volume', {}).get('level', 'normal')
        intensity = audio_features.get('characteristics', {}).get('intensity', 'normal')

        prompt_clips.append({
            "start": f"{start_time:.2f}",
            "end": f"{end_time:.2f}",
            "text": text,
            "volume": volume,
            "intensity": intensity
        })

    # Check if prompt_clips is empty after filtering/processing
    if not prompt_clips:
        print("Warning: No valid clips to send for ranking in this chunk.")
        return "" # Return empty string if no clips to rank

    prompt = f"""You are an expert content analyzer focusing on viral potential. Analyze these clips based ONLY on the provided data:
{json.dumps(prompt_clips, indent=2)}

For each clip, evaluate using:

1. Audio Engagement (40% weight):
- Use the provided 'volume' (quiet/normal/loud) and 'intensity' (normal/high) indicators. Higher volume/intensity generally suggests higher engagement potential.
- Consider variations if multiple clips are analyzed together (though typically done per clip).

2. Content Analysis (60% weight):
- Analyze the 'text' for controversial, quotable, or discussion-provoking content.
- Assess if the topic seems relevant or timely based *only* on the text provided. Avoid external knowledge.

For each clip analyzed, return ONLY valid JSON following this exact structure within a main "clips" list:

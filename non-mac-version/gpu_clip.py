# gpu_clip.py

from openai import OpenAI
import argparse
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

def setup_gpu():
    """Configure GPU settings if available."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        return True
    return False

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def load_clips(json_path: str) -> List[Dict]:
    """Load clip data from JSON file."""
    try:
        with open(json_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Clips file not found: {json_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {json_path}")

def process_chunk_gpu(chunk_data: Tuple[List[Dict], str, str, str, int]) -> List[Dict]:
    """Process a single chunk of clips."""
    clips, api_key, site_url, site_name, chunk_id = chunk_data
    
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        ranked_results = rank_clips_chunk(clips, api_key, site_url, site_name)
        if ranked_results:
            return parse_clip_data(ranked_results)
        return []
    except Exception as e:
        print(f"Warning: Failed to process chunk {chunk_id}: {str(e)}")
        return []

def rank_clips_chunk(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "") -> str:
    """Send clips to the API and receive rankings."""
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

For each clip, provide in this exact format:
1. **Clip Name: "[TITLE]"**
   Start: [START]s, End: [END]s
   Score: [1-10]
   Factors: [Key viral factors]
   Platforms: [Recommended platforms]

Rank clips by viral potential. Focus on measurable features in the data."""

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that ranks video clips. Keep explanations brief and focused on virality potential."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1,
                max_tokens=1000
            )
            if completion and completion.choices:
                return completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise RuntimeError(f"Failed to rank clips after {max_retries} attempts: {str(e)}")
    
    return None

def rank_all_clips_parallel(clips: List[Dict], api_key: str, site_url: str = "", site_name: str = "", 
                             chunk_size: int = 5, num_processes: int = None) -> List[Dict]:
    """Rank clips in parallel using multithreading."""
    if num_processes is None:
        num_processes = mp.cpu_count()

    chunks = chunk_list(clips, chunk_size)
    chunk_data = [(chunk, api_key, site_url, site_name, i) for i, chunk in enumerate(chunks)]
    
    all_ranked_clips = []
    pbar = tqdm(total=len(chunks), desc="Processing chunks")

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk_gpu, data) for data in chunk_data]
        for future in futures:
            try:
                result = future.result()
                all_ranked_clips.extend(result)
                pbar.update(1)
            except Exception as e:
                print(f"Warning: Chunk processing failed: {str(e)}")

    pbar.close()
    return sorted(all_ranked_clips, key=lambda x: x.get('score', 0), reverse=True)

def parse_clip_data(input_string: str) -> List[Dict]:
    """Parse the model's output into structured clip data."""
    if not input_string:
        return []

    clips = []
    current_clip = {}
    lines = input_string.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r'^\d+\.\s\*\*Clip Name:', line):
            if current_clip:
                clips.append(current_clip)
                current_clip = {}
            match = re.search(r'Clip Name: "(.*?)"', line)
            if match:
                current_clip['name'] = match.group(1)

        elif 'Start:' in line and 'End:' in line:
            match = re.search(r'Start: ([\d.]+)s, End: ([\d.]+)s', line)
            if match:
                current_clip['start'] = float(match.group(1))
                current_clip['end'] = float(match.group(2))

        elif 'Score:' in line:
            match = re.search(r'Score: (\d+)', line)
            if match:
                current_clip['score'] = int(match.group(1))

        elif 'Factors:' in line:
            match = re.search(r'Factors: (.+)', line)
            if match:
                current_clip['factors'] = match.group(1)

        elif 'Platforms:' in line:
            match = re.search(r'Platforms: (.+)', line)
            if match:
                current_clip['platforms'] = match.group(1)

    if current_clip:
        clips.append(current_clip)

    return clips

def save_top_clips_json(clips: List[Dict], output_file: str, num_clips: int = 20) -> None:
    """Save top clips to a JSON file."""
    top_clips = clips[:num_clips]
    output_data = {
        'top_clips': top_clips,
        'total_clips': len(clips),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Rank viral video clips using GPU acceleration and OpenRouter.')
    parser.add_argument('clips_json', help='Path to JSON file with clip metadata')
    parser.add_argument('--output_file', default='top_clips_one.json', help='Output JSON file')
    parser.add_argument('--site_url', default='http://localhost', help='Site URL for OpenRouter API')
    parser.add_argument('--site_name', default='Local Test', help='Site name for OpenRouter API')
    parser.add_argument('--num_clips', type=int, default=20, help='Number of top clips to output')
    parser.add_argument('--chunk_size', type=int, default=5, help='Number of clips per API call')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel threads')

    args = parser.parse_args()
    start_time = time.time()

    try:
        gpu_enabled = setup_gpu()
        print("GPU acceleration enabled" if gpu_enabled else "Warning: GPU not available, using CPU")

        api_key = os.getenv("OPEN_ROUTER_KEY")
        if not api_key:
            raise EnvironmentError("Please set the OPEN_ROUTER_KEY environment variable.")

        clips = load_clips(args.clips_json)
        ranked_clips = rank_all_clips_parallel(
            clips,
            api_key,
            args.site_url,
            args.site_name,
            args.chunk_size,
            args.num_processes
        )

        save_top_clips_json(ranked_clips, args.output_file, args.num_clips)
        print(f"\nSaved top {args.num_clips} clips to {args.output_file}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


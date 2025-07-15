# Try different ways to import Whisper for compatibility
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
import argparse
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
    audio_path = video_file.with_suffix('.wav')
    
    print(f"Extracting audio to {audio_path}...")
    
    subprocess.run([
        'ffmpeg', '-i', str(video_file), 
        '-vn', '-acodec', 'pcm_s16le', 
        '-ar', '16000', '-ac', '1', 
        str(audio_path)
    ])
    
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

def create_audio_chunks(audio_path, chunk_duration=300.0, overlap_duration=30.0):
    """Split audio into chunks with overlap for processing"""
    audio = AudioSegment.from_wav(str(audio_path))
    total_duration = len(audio) / 1000.0  # Convert to seconds
    
    chunks = []
    start_time = 0.0
    chunk_index = 0
    
    while start_time < total_duration:
        end_time = min(start_time + chunk_duration, total_duration)
        
        # Extract chunk with millisecond precision
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        chunk_audio = audio[start_ms:end_ms]
        
        # Save chunk to temporary file
        chunk_path = f"temp_chunk_{chunk_index}.wav"
        chunk_audio.export(chunk_path, format="wav")
        
        chunks.append({
            'path': chunk_path,
            'start_offset': start_time,
            'end_offset': end_time,
            'index': chunk_index
        })
        
        # Register for cleanup
        global files_to_cleanup
        files_to_cleanup.append(chunk_path)
        
        # Move start time forward (with overlap consideration)
        if end_time >= total_duration:
            break
        start_time = end_time - overlap_duration
        chunk_index += 1
    
    print(f"Created {len(chunks)} audio chunks of ~{chunk_duration/60:.1f} minutes each")
    return chunks

def merge_chunk_results(chunk_results, overlap_duration=30.0):
    """Merge transcription results from multiple chunks, handling overlaps"""
    if not chunk_results:
        return []
    
    merged_segments = []
    
    for i, chunk_result in enumerate(chunk_results):
        chunk_segments = chunk_result['segments']
        start_offset = chunk_result['start_offset']
        
        for segment in chunk_segments:
            # Adjust timestamps to global timeline
            adjusted_segment = segment.copy()
            adjusted_segment['start'] += start_offset
            adjusted_segment['end'] += start_offset
            
            # Skip overlapping segments from previous chunks
            if i > 0 and adjusted_segment['start'] < chunk_results[i-1]['end_offset'] - overlap_duration/2:
                continue
                
            merged_segments.append(adjusted_segment)
    
    return merged_segments

def transcribe_chunk(model, chunk_info, device, min_duration=30.0):
    """Transcribe a single audio chunk"""
    chunk_path = chunk_info['path']
    start_offset = chunk_info['start_offset']
    
    try:
        # Load chunk audio for feature extraction
        audio = AudioSegment.from_wav(chunk_path)
        
        # Transcribe chunk
        result = model.transcribe(chunk_path, language='en', fp16=(device == "cuda"))
        
        # Process segments with audio features
        enhanced_segments = []
        current_segments = []
        current_duration = 0.0
        
        for segment in result["segments"]:
            audio_features = extract_audio_features(
                audio,
                segment["start"],
                segment["end"]
            )
            
            enhanced_segment = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "audio_features": audio_features
            }
            
            current_segments.append(enhanced_segment)
            current_duration = current_segments[-1]["end"] - current_segments[0]["start"]
            
            if current_duration >= min_duration:
                combined_segment = combine_segments(current_segments)
                if combined_segment:
                    enhanced_segments.append(combined_segment)
                current_segments = []
                current_duration = 0.0
        
        # Handle remaining segments
        if current_segments:
            combined_segment = combine_segments(current_segments)
            if combined_segment:
                enhanced_segments.append(combined_segment)
        
        return {
            'segments': enhanced_segments,
            'start_offset': start_offset,
            'end_offset': chunk_info['end_offset'],
            'success': True
        }
        
    except Exception as e:
        print(f"Error processing chunk {chunk_info['index']}: {str(e)}")
        return {
            'segments': [],
            'start_offset': start_offset,
            'end_offset': chunk_info['end_offset'],
            'success': False,
            'error': str(e)
        }

def transcribe_with_features(model, audio_path, device, min_duration=30.0, chunk_duration=300.0):
    """Get transcription with timestamps and audio features using chunked processing"""
    print("Generating enhanced transcription with chunked processing...")
    
    transcribe_start = time.time()
    
    # Create audio chunks
    chunks = create_audio_chunks(audio_path, chunk_duration)
    
    # Process each chunk
    chunk_results = []
    failed_chunks = []
    
    for i, chunk_info in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} ({chunk_info['start_offset']:.1f}s - {chunk_info['end_offset']:.1f}s)")
        
        result = transcribe_chunk(model, chunk_info, device, min_duration)
        chunk_results.append(result)
        
        if not result['success']:
            failed_chunks.append(i)
            print(f"Warning: Chunk {i+1} failed: {result.get('error', 'Unknown error')}")
    
    # Report processing status
    if failed_chunks:
        print(f"Warning: {len(failed_chunks)} out of {len(chunks)} chunks failed")
        print(f"Failed chunks: {[i+1 for i in failed_chunks]}")
    
    # Merge results from successful chunks
    enhanced_segments = merge_chunk_results(chunk_results)
    
    transcribe_end = time.time()
    print(f"Enhanced transcription processing took: {format_time(transcribe_end - transcribe_start)}")
    print(f"Generated {len(enhanced_segments)} combined segments")
    
    return enhanced_segments

def cleanup_files():
    """Clean up temporary files created during processing"""
    global files_to_cleanup
    print("\nCleaning up temporary files...")
    for file_path in files_to_cleanup:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                print(f"Removed file: {file_path}")
        except Exception as e:
            print(f"Warning: Failed to remove {file_path}: {e}")

# Global list to track files for cleanup
files_to_cleanup = []

def process_video(video_path, model_size="base", chunk_duration=300.0):
    """Process video to create enhanced transcription"""
    process_start = time.time()
    device = check_gpu()
    
    video_file = Path(video_path)
    transcription_path = video_file.with_suffix('.enhanced_transcription.json')
    
    print(f"Processing {video_file.name}...")
    
    try:
        audio_path = extract_audio(video_path)
        
        print(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        if device == "cuda":
            model = model.cuda()
        
        enhanced_transcription = transcribe_with_features(model, audio_path, device, 
                                                        min_duration=30.0, 
                                                        chunk_duration=chunk_duration)
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_transcription, f, indent=2, ensure_ascii=False)
        
        process_end = time.time()
        print(f"Total processing time: {format_time(process_end - process_start)}")
        print(f"Enhanced transcription saved to {transcription_path}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise
    finally:
        # Clean up the audio file even if there was an error
        cleanup_files()

def main():
    parser = argparse.ArgumentParser(description='Create enhanced transcription with audio features')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--model', default='base', 
                      choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                      help='Whisper model size to use')
    parser.add_argument('--min-duration', type=float, default=30.0,
                      help='Minimum duration in seconds for combined segments')
    parser.add_argument('--chunk-duration', type=float, default=300.0,
                      help='Duration in seconds for audio chunks (default: 5 minutes)')
    
    args = parser.parse_args()
    
    # Register cleanup function to run at exit
    atexit.register(cleanup_files)
    
    try:
        process_video(args.video_path, model_size=args.model, chunk_duration=args.chunk_duration)
    except Exception as e:
        print(f"Failed to process video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

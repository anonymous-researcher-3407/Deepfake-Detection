#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
import subprocess
import math
import numpy as np
import tqdm
import argparse
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

# --- Parameters ---
MAX_THREADS = 64   # Keep low to prevent disk choking
SAMPLE_RATE = 16000
LABEL_RESOLUTION = 0.01  # 10ms
SEED = 42
MAX_CHUNK_DURATION = 15.0  # Maximum duration per chunk in seconds

# Set random seed
random.seed(SEED)
np.random.seed(SEED)

def get_args():
    parser = argparse.ArgumentParser(description="Prepare data for PartialSpoof")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the metadata JSON file")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing video files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--dataset_name", type=str, default="dev", help="Name of the dataset (e.g., dev, train)")
    parser.add_argument("--max_samples", type=int, default=40000000, help="Maximum number of samples to process")
    parser.add_argument("--from_idx", type=int, default=0, help="Start index for processing")
    parser.add_argument("--to_idx", type=int, default=1000000000000, help="End index for processing")
    return parser.parse_args()

def clear_output_paths(output_dir, dataset_name):
    """Clears and recreates all output directories and files."""
    print("Clearing output paths...")
    
    dev_con_wav = os.path.join(output_dir, dataset_name, 'con_wav')
    dev_lst = os.path.join(output_dir, dataset_name, f'{dataset_name}.lst')
    dev_seg_lab = os.path.join(output_dir, 'segment_labels', f'{dataset_name}_seglab_{LABEL_RESOLUTION}.npy')

    paths_to_process = [dev_con_wav, dev_lst, dev_seg_lab]

    for path in paths_to_process:
        dir_name = os.path.dirname(path)
        if not dir_name: dir_name = path
        if not os.path.exists(dir_name): os.makedirs(dir_name, exist_ok=True)

        if path.endswith('.npy') or path.endswith('.lst'):
            if os.path.exists(path): os.remove(path)
        else:
            if os.path.exists(path): shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    print("Output paths cleared.")
    return dev_con_wav, dev_lst, dev_seg_lab

def get_video_duration(video_path):
    """Uses ffprobe to get the exact duration of a video file."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def process_single_file(task):
    """
    Worker function to process one audio file.
    If audio is longer than MAX_CHUNK_DURATION, splits into equal-sized chunks.
    Each chunk is named with suffix _0, _1, _2, etc.
    Fake segment labels are adjusted for each chunk.
    """
    input_mp4_path, fake_segments, output_name, output_wav_folder = task

    try:
        # 1. Get total duration (needed for label array size)
        total_dur_sec = get_video_duration(input_mp4_path)
        
        if total_dur_sec <= 0:
            return ("MISSING_ZERO_DURATION", input_mp4_path)

        results = []  # List of (chunk_name, labels_array) tuples
        
        if total_dur_sec <= MAX_CHUNK_DURATION:
            # Short audio: process as single file (no chunking)
            output_wav_path = os.path.join(output_wav_folder, f"{output_name}.wav")
            
            cmd_ffmpeg = [
                'ffmpeg', '-v', 'error',
                '-threads', '1', '-y', 
                '-i', input_mp4_path,
                '-ar', str(SAMPLE_RATE),
                '-ac', '1',
                '-f', 'wav',
                output_wav_path
            ]
            subprocess.run(cmd_ffmpeg, check=True, timeout=120)

            # Generate labels for the full duration
            num_labels = int(math.ceil(total_dur_sec / LABEL_RESOLUTION))
            labels = np.ones(num_labels, dtype=int)

            for seg_start, seg_end in fake_segments:
                seg_start = max(0, seg_start)
                seg_end = min(total_dur_sec, seg_end)
                
                if seg_start < seg_end:
                    start_bin = int(math.floor(seg_start / LABEL_RESOLUTION))
                    end_bin = int(math.ceil(seg_end / LABEL_RESOLUTION))
                    start_bin = max(0, start_bin)
                    end_bin = min(num_labels, end_bin)
                    if start_bin < end_bin:
                        labels[start_bin:end_bin] = 0

            labels_str = labels.astype(str)
            results.append((output_name, labels_str))
        else:
            # Long audio: divide into equal-sized chunks
            # e.g., 16s -> 2 chunks of 8s each, not 15s + 1s
            num_chunks = int(math.ceil(total_dur_sec / MAX_CHUNK_DURATION))
            chunk_duration = total_dur_sec / num_chunks
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_duration
                chunk_end = min((chunk_idx + 1) * chunk_duration, total_dur_sec)
                chunk_dur = chunk_end - chunk_start
                
                # Chunk naming: originalname_0, originalname_1, etc.
                chunk_name = f"{output_name}_{chunk_idx}"
                output_wav_path = os.path.join(output_wav_folder, f"{chunk_name}.wav")
                
                # Extract chunk audio using ffmpeg with -ss and -t
                cmd_ffmpeg = [
                    'ffmpeg', '-v', 'error',
                    '-threads', '1', '-y',
                    '-ss', str(chunk_start),
                    '-i', input_mp4_path,
                    '-t', str(chunk_dur),
                    '-ar', str(SAMPLE_RATE),
                    '-ac', '1',
                    '-f', 'wav',
                    output_wav_path
                ]
                subprocess.run(cmd_ffmpeg, check=True, timeout=120)
                
                # Generate labels for this chunk
                num_labels = int(math.ceil(chunk_dur / LABEL_RESOLUTION))
                labels = np.ones(num_labels, dtype=int)
                
                # Adjust fake segments for this chunk (relative to chunk start)
                for seg_start, seg_end in fake_segments:
                    # Check if segment overlaps with this chunk
                    if seg_end <= chunk_start or seg_start >= chunk_end:
                        continue  # No overlap
                    
                    # Calculate positions relative to chunk start
                    rel_start = max(0, seg_start - chunk_start)
                    rel_end = min(chunk_dur, seg_end - chunk_start)
                    
                    if rel_start < rel_end:
                        start_bin = int(math.floor(rel_start / LABEL_RESOLUTION))
                        end_bin = int(math.ceil(rel_end / LABEL_RESOLUTION))
                        start_bin = max(0, start_bin)
                        end_bin = min(num_labels, end_bin)
                        if start_bin < end_bin:
                            labels[start_bin:end_bin] = 0
                
                labels_str = labels.astype(str)
                results.append((chunk_name, labels_str))
        
        return ("SUCCESS", results)

    except Exception as e:
        return ("MISSING_FFMPEG_CRASH", f"{input_mp4_path} | Error: {e}")

def process_dataset(from_idx, to_idx, json_path, video_folder, max_samples, output_wav_folder, output_lst_file, output_label_file, dataset_name):
    print(f"\n--- Starting processing for {dataset_name} ---")
    
    # 1. Load metadata
    print(f"Loading metadata from {json_path}...")
    with open(json_path, 'r') as f:
        all_data = json.load(f)
    
    all_data = all_data[from_idx:min(to_idx, len(all_data))]
    # 2. Build task list
    print(f"Building task list...")
    final_tasks = [] 
    
    for item in tqdm.tqdm(all_data, desc="Scanning metadata"):
        if len(final_tasks) >= max_samples:
            break

        # Get fake segments from both audio and visual (combine them)
        # A segment is fake if it's in either audio_fake_segments OR visual_fake_segments
        audio_fake_segments = item.get('audio_fake_segments', [])
        visual_fake_segments = item.get('visual_fake_segments', [])
        
        # Combine both lists
        fake_segments = audio_fake_segments # + visual_fake_segments
        
        # Path resolution
        relative_path = item['file']
        input_mp4_path = os.path.join(video_folder, relative_path)

        # Check existence (handle missing .mp4 extension logic)
        if not os.path.exists(input_mp4_path):
            path_with_ext = input_mp4_path + ".mp4"
            if os.path.exists(path_with_ext):
                input_mp4_path = path_with_ext
                relative_path = relative_path + ".mp4"
            else:
                continue
        
        # Output naming: Use the sanitized relative path
        base_name = relative_path.replace('.mp4', '').replace('/', '_')
        
        # Create Task: Pass output folder instead of specific path (for chunking support)
        task = (input_mp4_path, fake_segments, base_name, output_wav_folder)
        final_tasks.append(task)
    
    print(f"Built a task list of {len(final_tasks)} files.")

    # 3. Run multiprocessing
    print(f"Processing {len(final_tasks)} files...")
    segment_labels = defaultdict(list)
    file_list = []
    
    stats = {
        "SUCCESS": 0,  # Number of files successfully processed
        "CHUNKS": 0,   # Total number of chunks/segments created
        "MISSING_ZERO_DURATION": 0,
        "MISSING_FFMPEG_CRASH": 0
    }

    with ProcessPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(process_single_file, task) for task in final_tasks]
        
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(final_tasks), desc=f"Processing {dataset_name}"):
            result = future.result()
            status = result[0]
            
            if status == "SUCCESS":
                # Result format: ("SUCCESS", [(chunk_name, labels_array), ...])
                chunk_results = result[1]
                for chunk_name, labels_array in chunk_results:
                    segment_labels[chunk_name] = labels_array
                    file_list.append(chunk_name)
                    stats["CHUNKS"] += 1
                stats["SUCCESS"] += 1
            else:
                stats[status] += 1

    print("\n--- Processing Report ---")
    print(f"Total files requested: {len(final_tasks)}")
    print(f"Files processed successfully: {stats['SUCCESS']}")
    print(f"Total chunks/segments created: {stats['CHUNKS']}")
    print(f"Errors (Zero Dur): {stats['MISSING_ZERO_DURATION']}")
    print(f"Errors (Crash): {stats['MISSING_FFMPEG_CRASH']}")

    # 4. Save outputs
    print(f"Saving outputs for {dataset_name}...")
    file_list.sort()
    with open(output_lst_file, 'w') as f:
        for name in file_list:
            f.write(f"{name}\n")
    
    np.save(output_label_file, dict(segment_labels))
    print(f"--- Finished processing for {dataset_name} ---")

def main():
    args = get_args()
    
    output_wav_folder, output_lst_file, output_label_file = clear_output_paths(args.output_dir, args.dataset_name)
    
    process_dataset(
        from_idx=args.from_idx,
        to_idx=args.to_idx,
        json_path=args.json_path,
        video_folder=args.video_folder,
        max_samples=args.max_samples,
        output_wav_folder=output_wav_folder,
        output_lst_file=output_lst_file,
        output_label_file=output_label_file,
        dataset_name=args.dataset_name
    )

    print("\nAll data preparation complete.")

if __name__ == "__main__":
    main()
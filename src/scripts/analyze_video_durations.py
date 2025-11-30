#!/usr/bin/env python3
"""
Analyze video durations and file sizes to find statistics and distributions.
"""

import sys
from pathlib import Path
import polars as pl
import av
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.data import load_metadata
from lib.utils.paths import resolve_video_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        container.close()
        return duration
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}", file=sys.stderr)
        return None


def get_video_file_size(video_path: str) -> int:
    """Get video file size in bytes."""
    try:
        return Path(video_path).stat().st_size
    except Exception as e:
        print(f"Error getting file size for {video_path}: {e}", file=sys.stderr)
        return None


def format_bytes(bytes_size: int) -> str:
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def main():
    project_root_path = Path(project_root)
    
    # Load metadata
    input_metadata_path = None
    for csv_name in ["FVC_dup.csv", "video_index_input.csv"]:
        candidate_path = project_root_path / "data" / csv_name
        if candidate_path.exists():
            input_metadata_path = candidate_path
            break
    
    if not input_metadata_path:
        print("Error: Cannot find metadata file")
        return
    
    df = load_metadata(str(input_metadata_path))
    if 'is_original' in df.columns:
        df = df.filter(pl.col('is_original') == True)
    
    print(f"Analyzing {df.height} videos...")
    print("=" * 80)
    
    durations = []
    file_sizes = []
    video_info = []  # Store (duration, file_size, video_path) tuples
    
    for row in df.iter_rows(named=True):
        video_rel = row.get('video_path', '')
        try:
            video_path = resolve_video_path(video_rel, project_root_path)
            if Path(video_path).exists():
                duration = get_video_duration(video_path)
                file_size = get_video_file_size(video_path)
                
                if duration is not None:
                    durations.append(duration)
                    video_info.append((duration, file_size, video_rel))
                
                if file_size is not None:
                    file_sizes.append(file_size)
        except Exception as e:
            print(f"Error processing {video_rel}: {e}", file=sys.stderr)
    
    if not durations:
        print("No durations found")
        return
    
    # Sort durations for median calculation
    durations_sorted = sorted(durations)
    median = durations_sorted[len(durations_sorted) // 2]
    max_duration = max(durations)
    min_duration = min(durations)
    mean_duration = sum(durations) / len(durations)
    
    # File size statistics
    if file_sizes:
        file_sizes_sorted = sorted(file_sizes)
        median_size = file_sizes_sorted[len(file_sizes_sorted) // 2]
        max_size = max(file_sizes)
        min_size = min(file_sizes)
        mean_size = sum(file_sizes) / len(file_sizes)
        total_size = sum(file_sizes)
    else:
        median_size = max_size = min_size = mean_size = total_size = 0
    
    print(f"\nVideo Duration Statistics:")
    print(f"  Total videos analyzed: {len(durations)}")
    print(f"  Min duration: {min_duration:.2f} seconds ({min_duration/60:.2f} minutes)")
    print(f"  Max duration: {max_duration:.2f} seconds ({max_duration/60:.2f} minutes)")
    print(f"  Median duration: {median:.2f} seconds ({median/60:.2f} minutes)")
    print(f"  Mean duration: {mean_duration:.2f} seconds ({mean_duration/60:.2f} minutes)")
    
    print(f"\nVideo File Size Statistics:")
    print(f"  Total videos with size info: {len(file_sizes)}")
    if file_sizes:
        print(f"  Min size: {format_bytes(min_size)}")
        print(f"  Max size: {format_bytes(max_size)}")
        print(f"  Median size: {format_bytes(median_size)}")
        print(f"  Mean size: {format_bytes(mean_size)}")
        print(f"  Total size: {format_bytes(total_size)}")
    
    # 30-second binning
    print(f"\n" + "=" * 80)
    print("30-Second Duration Binning:")
    print("=" * 80)
    
    # Create bins: [0, 30), [30, 60), [60, 90), etc.
    bin_counts = defaultdict(int)
    bin_videos = defaultdict(list)  # Store video paths in each bin
    
    for duration, file_size, video_rel in video_info:
        bin_start = int(duration // 30) * 30
        bin_end = bin_start + 30
        bin_label = f"[{bin_start},{bin_end})"
        bin_counts[bin_label] += 1
        bin_videos[bin_label].append((duration, file_size, video_rel))
    
    # Sort bins by start time
    sorted_bins = sorted(bin_counts.keys(), key=lambda x: int(x.split(',')[0][1:]))
    
    print(f"{'Bin (seconds)':<20} {'Count':<10} {'Percentage':<12} {'Avg Size':<15}")
    print("-" * 80)
    
    for bin_label in sorted_bins:
        count = bin_counts[bin_label]
        percentage = (count / len(durations)) * 100
        
        # Calculate average file size for this bin
        bin_video_list = bin_videos[bin_label]
        if bin_video_list:
            avg_size = sum(size for _, size, _ in bin_video_list if size is not None) / len([s for _, s, _ in bin_video_list if s is not None])
            avg_size_str = format_bytes(int(avg_size)) if avg_size > 0 else "N/A"
        else:
            avg_size_str = "N/A"
        
        print(f"{bin_label:<20} {count:<10} {percentage:>6.2f}%     {avg_size_str:<15}")
    
    # Summary statistics
    print(f"\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    
    # Check how many videos exceed 1000 frames at 30fps (33.33 seconds)
    videos_over_33_sec = sum(1 for d in durations if d > 33.33)
    videos_over_60_sec = sum(1 for d in durations if d > 60.0)
    videos_over_120_sec = sum(1 for d in durations if d > 120.0)
    
    print(f"Videos exceeding 33.33 seconds (1000 frames @ 30fps): {videos_over_33_sec} ({videos_over_33_sec/len(durations)*100:.1f}%)")
    print(f"Videos exceeding 60 seconds: {videos_over_60_sec} ({videos_over_60_sec/len(durations)*100:.1f}%)")
    print(f"Videos exceeding 120 seconds: {videos_over_120_sec} ({videos_over_120_sec/len(durations)*100:.1f}%)")
    
    # Show largest videos by size
    if file_sizes:
        print(f"\nTop 5 Largest Videos by File Size:")
        video_info_with_size = [(d, s, v) for d, s, v in video_info if s is not None]
        video_info_with_size.sort(key=lambda x: x[1], reverse=True)
        for i, (duration, size, video_rel) in enumerate(video_info_with_size[:5], 1):
            print(f"  {i}. {Path(video_rel).name}: {format_bytes(size)} ({duration:.2f}s)")
    
    # Show longest videos
    print(f"\nTop 5 Longest Videos by Duration:")
    video_info_sorted_by_duration = sorted(video_info, key=lambda x: x[0], reverse=True)
    for i, (duration, size, video_rel) in enumerate(video_info_sorted_by_duration[:5], 1):
        size_str = format_bytes(size) if size is not None else "N/A"
        print(f"  {i}. {Path(video_rel).name}: {duration:.2f}s ({duration/60:.2f} min) - {size_str}")


if __name__ == "__main__":
    main()


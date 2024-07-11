"""Gopro Sync Tool for sychronization two gopro videos using an impulse noise
Usage: python goprosync.py --left PATH_TO_LEFT_MP4 
        --right PATH_TO_RIGHT_MP4 --out PATH_TO_OUTPUT_DIRECTORY
"""
import sys
from pathlib import Path
import argparse
import shutil

import wave
import numpy as np
import stumpy
from moviepy.editor import VideoFileClip

def gopro_sync(left: Path, right: Path, trimmed_left: Path, trimmed_right: Path):
    """Perform gopro synchronization.

    Keyword arguments:
    left -- path to the left camera mp4
    right -- path to the right camera mp4
    trimmed_left -- output path for the trimmed left image
    trimmed_right -- output path for the trimmed right image
    """

    # Create a temporary directory to store the audio files
    curr_directory = Path('./.temp')
    curr_directory.mkdir(exist_ok=True)

    left_audio_path = curr_directory.joinpath("left.wav")
    right_audio_path = curr_directory.joinpath("right.wav")

    left_video = VideoFileClip(left.as_posix())
    left_video.audio.write_audiofile(left_audio_path)
    left_video.close()

    right_video = VideoFileClip(right.as_posix())
    right_video.audio.write_audiofile(right_audio_path)
    right_video.close()

    # Begin uploading audio from file into a numpy array

    left_wav = wave.open(left_audio_path.as_posix(), "rb")
    right_wav = wave.open(right_audio_path.as_posix(), "rb")

    left_freq = left_wav.getframerate()
    right_freq = right_wav.getframerate()

    left_samples = left_wav.getnframes()
    right_samples = right_wav.getnframes()

    left_signal = left_wav.readframes(left_samples)
    right_signal = right_wav.readframes(right_samples)

    left_signal_array = np.frombuffer(left_signal, dtype=np.int16)
    right_signal_array = np.frombuffer(right_signal, dtype=np.int16)

    # Remove audio files created for temporary use.
    left_wav.close()
    right_wav.close()

    # Delete the temporary directory
    shutil.rmtree(curr_directory)

    # Continue off from before, creating channels and times 

    times_left = np.linspace(0, left_samples/left_freq, num=left_samples)
    times_right = np.linspace(0, right_samples/right_freq, num=right_samples)

    # Use right channel for left camera and left channel for right camera
    left_channel = left_signal_array[1::2]
    right_channel = right_signal_array[0::2]

    # Find the impulse. The impulse should be the highest value in the signal.
    # Use the region around the impulse for matrix profile later

    left_spike = left_channel.argmax()
    right_spike = right_channel.argmax()

    left_spike_start = left_spike-10000
    left_spike_end = left_spike+10000

    right_spike_start = right_spike-10000
    right_spike_end = right_spike+10000

    # Prepare the subarrays for the matrix profile algorithm
    # The matrix profile algorithm, specifically for conserved pattern detection
    # To read more about matrix profiles, visit matrixprofile.org

    m = 10000
    left_subarray = left_channel[left_spike_start:left_spike_end]
    right_subarray = right_channel[right_spike_start:right_spike_end]

    # Matrix profile can find patterns in nlogn time, 
    # so we use it to quickly find conserved patterns.
    matrix_profile=stumpy.stump(T_A=left_subarray.astype(np.float64),
                                  m=m, T_B=right_subarray.astype(np.float64), ignore_trivial=False)

    # Find the index of the left and right motifs
    left_motif_index = matrix_profile[:,0].argmin()
    right_motif_index = matrix_profile[left_motif_index,1] + right_spike_start
    left_motif_index += left_spike_start    

    # Extract the desired cropped left camera video
    left_video = VideoFileClip(left.as_posix())
    left_video_trimmed = left_video.subclip(times_left[left_motif_index],times_left[-1])
    left_video_trimmed.write_videofile(trimmed_left.as_posix())
    left_video.close()
    left_video_trimmed.close()

    right_video = VideoFileClip(right.as_posix())
    right_video_trimmed = right_video.subclip(times_right[right_motif_index],times_right[-1])
    right_video_trimmed.write_videofile(trimmed_right.as_posix())
    right_video.close()
    right_video_trimmed.close()


def main():
    """Check arguments and set up paths, then call goprosync function
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)

    args = parser.parse_args()

    # Make sure that all the paths are valid

    if not args.left.is_file():
        print("Left path is not a file.")
        sys.exit(1)

    elif not args.right.is_file():
        print("Right path is not file")
        sys.exit(1)

    # Create output directory or use provided one if exists

    args.out.mkdir(exist_ok=True)

    # Make sure that input files are mp4s

    if args.left.suffix not in (".mp4", ".MP4"):
        print("Ensure that left file is an mp4")
        sys.exit(1)

    if args.right.suffix not in (".mp4", ".MP4"):
        print("Ensure that right file is an mp4")
        sys.exit(1)

    left_trimmed = args.out.joinpath("left_trimmed.mp4")
    right_trimmed = args.out.joinpath("right_trimmed.mp4")

    gopro_sync(left=args.left, right=args.right, 
               trimmed_left=left_trimmed, trimmed_right=right_trimmed)

if __name__ == '__main__':
    main()

import wave
import numpy as np
import matplotlib.pyplot as plt
import stumpy
import os
import sys
from moviepy.editor import VideoFileClip 
import tempfile
from pathlib import Path

def gopro_sync(left, right, trimmed_left, trimmed_right):

    left = Path(left)
    right = Path(right)
    trimmed_left = Path(trimmed_left)
    trimmed_right = Path(trimmed_right)

    # Create a temporary directory to store the audio files
    
    try:
        os.mkdir(".temp")
    except FileExistsError:
        pass
    
    curr_directory = Path(os.getcwd()).joinpath(".temp")

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
    
    Path.unlink(left_audio_path)
    Path.unlink(right_audio_path)
    
    # Only delete the directory if it's not already empty
    try:
        Path.rmdir(curr_directory)
    except Exception:
        pass

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

    # Matrix profile can find patterns in nlogn time, so we use it to quickly find conserved patterns.
    matrix_profile = stumpy.stump(T_A=left_subarray.astype(np.float64), m=m, T_B=right_subarray.astype(np.float64), ignore_trivial=False)

    # Find the index of the left and right motifs
    left_motif_index = matrix_profile[:,0].argmin()
    right_motif_index = matrix_profile[left_motif_index,1] + right_spike_start
    left_motif_index += left_spike_start    

    # Extract the desired cropped left camera video
    left_video = VideoFileClip(left.as_posix())
    left_video_trimmed = left_video.subclip(times_left[left_motif_index],times_left[-1])
    left_video_trimmed.write_videofile(trimmed_left.as_posix())

    right_video = VideoFileClip(right.as_posix())
    right_video_trimmed = right_video.subclip(times_right[right_motif_index],times_right[-1])
    right_video_trimmed.write_videofile(trimmed_right.as_posix())


def argument_parser(args):

    # make sure flags are used correctly
    acceptable_args = ["--left", "--out", "--right"]
    args_1 = args[1:3]
    args_2 = args[3:5]
    args_3 = args[5:7]

    if args_1[0] not in acceptable_args:
        print(f"{args_1[0]} is not an acceptable flag. Please use the flags --left, --right, --out")
        exit()
    elif args_2[0] not in acceptable_args:
        print(f"{args_2[0]} is not an acceptable flag. Please use the flags --left, --right, --out")
        exit()
    elif args_3[0] not in acceptable_args:
        print(f"{args_3[0]} is not an acceptable flag. Please use the flags --left, --right, --out")
        exit()

    if len(set([args_1[0], args_2[0], args_3[0]])) != 3:
        print("Duplicate flags. Please try again")
        exit()

    # Getting all the paths organized into the appropriate variables
    path_left, path_right, path_out = ("", "", "")

    if (args_1[0] == "--left"):
        path_left = args_1[1]
        if (args_2[0] == "--right"):
            path_right = args_2[1]
            path_out = args_3[1]
        else:
            path_out = args_2[1]
            path_right = args_3[1]

    elif (args_1[0] == "--right"):
        path_right = args_1[1]
        if (args_2[0] == "--left"):
            path_left = args_2[1]
            path_out = args_3[1]
        else:
            path_out = args_2[1]
            path_left = args_3[1]
    
    elif (args_1[0] == "--out"):
        path_out = args_1[1]
        if (args_2[0] == "--left"):
            path_left = args_2[1]
            path_right = args_3[1]
        else:
            path_right = args_2[1]
            path_left = args_3[1]        

    # Ensure paths exist
    if not os.path.isfile(path_left):
        print(f"{path_left} is not an acceptable path for flag --left, which should point to a file.")
        exit()
    elif not os.path.isfile(path_right):
        print(f"{path_right} is not an acceptable path for flag --right, which should point to a file.")
        exit()
    elif not os.path.isdir(path_out):
        print(f"{path_out} is not an acceptable path for flag --out, which should point to a directory.")
        exit()

    # Ensure that left and right files are mp4s
    if (path_left[-4:] != ".mp4" and path_left[-4:] != ".MP4"):
        print("File for left is not an mp4. Please try again.")
        exit()
    elif (path_right[-4:] != ".mp4" and path_right[-4:] != ".MP4"):
        print("File for right is not an mp4. Please try again.")
        exit()

    return path_left, path_right, path_out

# Check number of arguments is correct
total_arguments = len(sys.argv)
if (total_arguments != 7):

    print("Incorrect number of arguments. Please ensure there are 6 arguments after the file name. For example: ")
    print("python3 goprosync.py --left left.mp4 --right right.mp4 --out ./final/")
    exit()

# Ensure that all arguments are made correctly
left_path, right_path, out_path = argument_parser(sys.argv)
left_trimmed = Path(out_path, "left_trimmed.mp4")
right_trimmed = Path(out_path, "right_trimmed.mp4")

gopro_sync(left=left_path, right=right_path, trimmed_left=left_trimmed, trimmed_right=right_trimmed)





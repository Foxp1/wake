import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import subprocess
from pydub import AudioSegment
import numpy as np
import sounddevice as sd
import soundfile as sf
from ipywebrtc import AudioRecorder, CameraStream
import time
import pyaudio


# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# Retrieve and plot spectogram of wav file
def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim

    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx


# Find segment time interval in 10s clip
def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


# Check if segment temporally overlaps with previous segments
def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time

    # Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments: # @KEEP
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break

    return overlap


# Overlay background with an audio clip
def insert_audio_clip(background, audio_clip, previous_segments):
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    # Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)

    # Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap. To avoid an endless loop
    # we retry 5 times(≈ 2 lines)
    retry = 5 # @KEEP
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1
    # if last try is not overlaping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):
        # Append the new segment_time to the list of previous_segments (≈ 1 line)
        previous_segments.append(segment_time)
        # Superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])
    else:
        new_background = background
        segment_time = (10000, 10000)

    return new_background, segment_time


# Update label vector. Adds 50 ones after the end of the wake word
def insert_ones(y, segment_end_ms):
    _, Ty = y.shape

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    if segment_end_y < Ty:
        # Add 1 to the correct index in the background label (y)
        for i in range(segment_end_y+1, segment_end_y+51):
            if i < Ty:
                y[0, i] = 1

    return y


# Create a training example given background, wake words, and negative words. Returns
# the spectrogram x and ground truths y
def create_training_example(background, activates, negatives, Ty, save_flag=0):
    # Make background quieter
    background = background - 20

    # Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1,Ty))

    # Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates: # @KEEP
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y" at segment_end
        y = insert_ones(y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives: # @KEEP
        # Insert the audio clip on the background
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # Export new training example
    file_handle = background.export("train" + ".wav", format="wav")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    if save_flag:
        k = 1
        for filename in os.listdir(save_flag):
            if filename.endswith("wav"):
                k+=1
        background.export(save_flag + str(k) + ".wav", format="wav")

    return x, y


# Compute the probabiliy of the wake word being said in the given file
def detect_wakeword(model, filename):
    plt.subplot(2, 1, 1)

    # Correct the amplitude of the input file before prediction
    audio_clip = AudioSegment.from_wav(filename)
    audio_clip = match_target_amplitude(audio_clip, -20.0)
    file_handle = audio_clip.export("tmp.wav", format="wav")
    filename = "tmp.wav"

    x = graph_spectrogram(filename)
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)

    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

# Insert a bell after wake word and save file in bell_output.wav
def bell_on_activate(filename, predictions, threshold):
    bell_file = "audio/bell/bell.wav"
    audio_clip = AudioSegment.from_wav(filename)
    bell = AudioSegment.from_wav(bell_file)
    Ty = predictions.shape[1]
    # Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Loop over the output steps in the y
    for i in range(Ty):
        # Increment consecutive output steps
        consecutive_timesteps += 1
        # If prediction is higher than the threshold and more than 20 consecutive output steps have passed
        if consecutive_timesteps > 20:
            # Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(bell, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Reset consecutive output steps to 0
            consecutive_timesteps = 0
        # if amplitude is smaller than the threshold reset the consecutive_timesteps counter
        if predictions[0, i, 0] < threshold:
            consecutive_timesteps = 0

    audio_clip.export("bell_output.wav", format='wav')


# Record audio sample using 'sounddevice'
def recordsample(path, duration, samplerate):
    k = 1
    for filename in os.listdir(path):
        if filename.endswith("wav"):
            k += 1
    filename = path + str(k) + ".wav"
    print('Recording started...')
    record = sd.rec(int(samplerate*duration), samplerate, channels=1, blocking = True)
    print('End')
    sd.wait()
    sf.write(filename, record, samplerate)

# Record sample using 'ipywebrtc'
def savesample(recorder, path):
    with open('recording.webm', 'wb') as f:
        f.write(recorder.audio.value)

    k = 1
    for filename in os.listdir(path):
        if filename.endswith("wav"):
            k += 1
    fullpath = path + "s_" + str(k) + ".wav"
    subprocess.run(["ffmpeg", "-i", "recording.webm", "-ac", "1", "-ar", "44100", "-f", "wav", fullpath, "-y"])

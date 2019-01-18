
# coding: utf-8

# In[1]:


import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2


# In[3]:


# load a wave data
# waveファイルのロード処理
def load_wave_data(train_dir, file_name):
    file_path = os.path.join(train_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x,fs

# change wave data to mel-stft
# waveファイルの波形データをメルスペクトログラム画像に変換
def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft,n_mels=128)
    return melsp

# display wave in plots
# waveファイルの波形データを表示
def show_wave(x):
    plt.plot(x)
    plt.show()

# display wave in heatmap
# メルスペクトログラムの画像表示
def show_melsp(melsp, fs):
    librosa.display.specshow(melsp, sr=fs)
    plt.colorbar()
    plt.show()

# data augmentation: add white noise
# ホワイトノイズを混ぜ込んだ波形データを生成
def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
# タイムシフトした波形データを生成
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
# ストレッチサウンドした波形データを生成
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")

# save wave data in npz, with augmentation
def save_np_data(filename, x, y, y_start_index, y_end_index, x_data_size, aug=None, rates=None):
    np_data = np.zeros(freq*time*x_data_size).reshape(x_data_size, freq, time)
    np_targets = np.zeros(y_end_index - y_start_index)
    index = 0
    for i in range(y_start_index, y_end_index):
        _x, fs = Common.load_wave_data(train_dir, x[i])
        if aug is not None:
            _x = aug(x=_x, rate=rates[i])
        _x = Common.calculate_melsp(_x)
        # resizeで足りない秒数は0埋めにする
        res = cv2.resize(_x, dsize=(time, freq), interpolation=cv2.INTER_CUBIC)
        np_data[index] = res
        np_targets[index] = y[i]
        index += 1
    np.savez(filename, x=np_data, y=np_targets)


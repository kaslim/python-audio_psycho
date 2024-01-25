import librosa
import numpy as np
from psyacmodel import mapping2barkmat, mapping2bark

# 加载音频文件
audio_path = '/Users/yuxuanliu/PycharmProjects/Python-Audio-Coder/PythonPsychoacoustics/output/The_Latin_Fab_Four-Yesterday.wav'
x, fs = librosa.load(audio_path, sr=None)  # 保留原始采样率

# STFT参数
n_fft = 2048  # FFT窗口大小
hop_length = n_fft // 2  # 帧之间的重叠

# 心理声学模型参数
nfilts = 64  # 巴克标度子带数量

# 初始化映射矩阵
W = mapping2barkmat(fs, nfilts, n_fft)

# 计算STFT
S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

# 初始化用于保存巴克尺度数值的列表
bark_scale_values = []

# 处理每个STFT窗口
for i in range(S.shape[1]):
    stft_frame = S[:, i]
    mX = np.abs(stft_frame)
    mXbark = mapping2bark(mX, W, n_fft)
    bark_scale_values.append(mXbark)

# 输出巴克尺度数值
print("Bark scale values for each window:")
for i, values in enumerate(bark_scale_values):
    print(f"Window {i}: {values}")
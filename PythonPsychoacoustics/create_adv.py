from psyacmodel import *

if __name__ == '__main__':
    # testing:
    import matplotlib.pyplot as plt
    import sound

    fs = 32000  # sampling frequency of audio signal
    maxfreq = fs / 2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfilts = 64  # number of subbands in the bark domain
    nfft = 2048  # number of fft subbands

    W = mapping2barkmat(fs, nfilts, nfft)
    plt.imshow(W[:, :256], cmap='Blues')
    # plt.imshow(W[:,], cmap='Blues')
    plt.title('Matrix W for Uniform to Bark Mapping as Image')
    plt.xlabel('Uniform Subbands')
    plt.ylabel('Bark Subbands')
    plt.show()

    W_inv = mappingfrombarkmat(W, nfft)
    plt.imshow(W_inv[:256, :], cmap='Blues')
    plt.title('Matrix W_inv for Bark to Uniform Mapping as Image')
    plt.xlabel('Bark Subbands')
    plt.ylabel('Uniform Subbands')
    plt.show()

    spreadingfunctionBarkdB = f_SP_dB(maxfreq, nfilts)
    print("spread-----", spreadingfunctionBarkdB)
    # x-axis: maxbark Bark in nfilts steps:
    maxbark = hz2bark(maxfreq)
    print("maxfreq=", maxfreq, "maxbark=", maxbark)
    bark = np.linspace(0, maxbark, nfilts)
    # The prototype over "nfilt" bands or 22 Bark, its center
    # shifted down to 22-26/nfilts*22=13 Bark:
    plt.plot(bark, spreadingfunctionBarkdB[26:(26 + nfilts)])
    plt.axis([6, 23, -100, 0])
    plt.xlabel('Bark')
    plt.ylabel('dB')
    plt.title('Spreading Function')
    plt.show()

    spreadingfuncmatrix = spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts)
    plt.imshow(spreadingfuncmatrix)
    plt.title('Matrix spreadingfuncmatrix as Image')
    plt.xlabel('Bark Domain Subbands')
    plt.ylabel('Bark Domain Subbands')
    plt.show()

    # -Testing-----------------------------------------
    # A test magnitude spectrum:
    # White noise:
    """
  if torch.cuda.is_available():
    device = 'cuda:0'
  else:
    device = 'cpu'

  # 加载音频
  x_np, sr = librosa.load("/Users/yuxuanliu/Downloads/move_new/data/The_Latin_Fab_Four-Yesterday.wav", sr=None)  # sr=None 保留原始的采样率
  x = torch.tensor(x_np).float().to(device)
  """

    import librosa
    import numpy as np

    # 加载音频文件
    audio_path = '/Users/yuxuanliu/PycharmProjects/Python-Audio-Coder/PythonPsychoacoustics/output/The_Latin_Fab_Four-Yesterday.wav'
    x, fs = librosa.load(audio_path, sr=None)  # 保留原始采样率

    # STFT参数
    n_fft = 2048  # FFT窗口大小
    hop_length = n_fft // 2  # 帧之间的重叠

    total_samples = len(x)  # 音频文件的总样本数

    # 计算傅立叶窗口的数量
    num_windows = np.ceil((total_samples - n_fft) / hop_length) + 1

    # 打印窗口数量
    print(f"Number of Fourier windows: {int(num_windows)}")

    # 心理声学模型参数
    nfilts = 64  # 巴克标度子带数量
    alpha = 0.8  # 非线性叠加指数

    # 初始化映射矩阵和扩展函数
    W = mapping2barkmat(fs, nfilts, n_fft)
    W_inv = mappingfrombarkmat(W, n_fft)
    spreadingfunctionBarkdB = f_SP_dB(fs / 2, nfilts)
    spreadingfuncmatrix = spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts)

    # 计算STFT
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

    # 初始化用于保存遮蔽阈值的矩阵
    mT_matrix = np.zeros_like(S, dtype=np.float32)

    # 处理每个STFT窗口
    for i in range(S.shape[1]):
        stft_frame = S[:, i]
        mX = np.abs(stft_frame)  # 幅度
        mXbark = mapping2bark(mX, W, n_fft)  # bark尺度
        mTbark = maskingThresholdBark(mXbark, spreadingfuncmatrix, alpha, fs, nfilts)  # bark
        mT = mappingfrombark(mTbark, W_inv, n_fft)
        mT_matrix[:, i] = mT

        # 处理 mT（例如保存或可视化）

    for index, value in enumerate(mT_matrix[:, 1]):
        print(f"Index {index}: Value {value}")

    print("length", len(mT_matrix[:, 1]))

    plt.figure(figsize=(10, 4))
    print("changdu: ", mT_matrix.shape[1])
    librosa.display.specshow(librosa.amplitude_to_db(mT_matrix, ref=np.max),
                             sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Masking Threshold over Time')
    plt.tight_layout()
    plt.show()

    """
      以下代码为尝试生成最大阈值限制下的最大噪声

    # 1. 生成噪声
    noise = np.random.randn(*S.shape)

    # 2. 调整噪声以符合遮蔽阈值
    noise_adjusted = np.zeros_like(S)
    for i in range(S.shape[1]):  # 遍历所有时间窗口
        # 获取当前窗口的遮蔽阈值
        mT_current = mT_matrix[:, i]
        # 将噪声幅度限制在遮蔽阈值以下
        noise_magnitude = np.minimum(np.abs(noise[:, i]), mT_current)
        # 保持噪声的相位
        noise_phase = np.angle(noise[:, i])
        # 构造受限制的噪声
        noise_adjusted[:, i] = noise_magnitude * np.exp(1j * noise_phase)

    # noise_adjusted = noise_adjusted * 2
    # 3. 将受限制的噪声添加到原始信号的STFT
    S_noisy = S + noise_adjusted
    """

    # 假设 S 是原始的STFT矩阵，mT_matrix 是掩蔽阈值矩阵
    # 生成与 S 形状相同的噪声矩阵
    noise = np.random.randn(*S.shape) + 1j * np.random.randn(*S.shape)
    # noise = np.random.randn(*S.shape)  # 对于噪声不添加相位信息

    # 调整噪声的幅度，使其等于掩蔽阈值
    noise_adjusted = np.zeros_like(S)
    for i in range(S.shape[1]):  # 遍历所有时间窗口
        # 获取当前窗口的掩蔽阈值
        mT_current = mT_matrix[:, i]

        # 将噪声幅度设置为掩蔽阈值
        noise_magnitude = mT_current

        # 保持噪声的相位
        noise_phase = np.angle(noise[:, i])

        # 构造与掩蔽阈值等值的噪声
        noise_adjusted_signal = noise_magnitude * np.exp(1j * noise_phase)

        # 确保第一个频带不添加噪声
        noise_adjusted_signal[0] = 0

        # 应用修改后的噪声
        noise_adjusted[:, i] = noise_adjusted_signal

    # 将调整后的噪声添加到原始信号的STFT
    S_noisy = S + noise_adjusted

    # 4. 逆STFT重建信号
    x_noisy = librosa.istft(S_noisy, hop_length=hop_length)
    x_only_noise = librosa.istft(noise_adjusted, hop_length=hop_length)

    # 5. 导出音频文件
    output_path = './output/noisy_Yesterday_method_2.wav'
    sf.write(output_path, x_noisy, fs)
    # 仅仅输出噪声
    noise_output_path = './output/only_noise_Yesterday_method_2.wav'
    sf.write(noise_output_path, x_only_noise, fs)
    print(f"Added noise audio file saved as {output_path}")

    # 计算信号的功率
    signal_power = np.mean(x ** 2)

    # 计算噪声的功率
    noise_power = np.mean(x_only_noise ** 2)

    # 计算信噪比
    SNR = 10 * np.log10(signal_power / noise_power)

    print(f"Signal-to-Noise Ratio (SNR): {SNR} dB")

# Programs to implement a psycho-acoustic model
# Using a matrix for the spreading function (faster)
# Gerald Schuller, Nov. 2016

import numpy as np
import torch
import librosa
import soundfile as sf


def f_SP_dB(maxfreq, nfilts):
    # usage: spreadingfunctionmatdB=f_SP_dB(maxfreq,nfilts)
    # computes the spreading function protoype, in the Bark scale.
    # Arguments: maxfreq: half the sampling freqency
    # nfilts: Number of subbands in the Bark domain, for instance 64
    maxbark = hz2bark(maxfreq)  # upper end of our Bark scale:22 Bark at 16 kHz
    # Number of our Bark scale bands over this range: nfilts=64
    spreadingfunctionBarkdB = np.zeros(2 * nfilts)
    # Spreading function prototype, "nfilts" bands for lower slope
    spreadingfunctionBarkdB[0:nfilts] = np.linspace(-maxbark * 27, -8, nfilts) - 23.5
    # "nfilts" bands for upper slope:
    spreadingfunctionBarkdB[nfilts:2 * nfilts] = np.linspace(0, -maxbark * 12.0, nfilts) - 23.5
    return spreadingfunctionBarkdB


def spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts):
    # Turns the spreading prototype function into a matrix of shifted versions.
    # Convert from dB to "voltage" and include alpha exponent
    # nfilts: Number of subbands in the Bark domain, for instance 64
    spreadingfunctionBarkVoltage = 10.0 ** (spreadingfunctionBarkdB / 20.0 * alpha)
    # Spreading functions for all bark scale bands in a matrix:
    spreadingfuncmatrix = np.zeros((nfilts, nfilts))
    for k in range(nfilts):
        spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[(nfilts - k):(2 * nfilts - k)]
    return spreadingfuncmatrix


def maskingThresholdBark(mXbark, spreadingfuncmatrix, alpha, fs, nfilts):
    # Computes the masking threshold on the Bark scale with non-linear superposition
    # usage: mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha)
    # Arg: mXbark: magnitude of FFT spectrum, on the Bark scale
    # spreadingfuncmatrix: spreading function matrix from function spreadingfunctionmat
    # alpha: exponent for non-linear superposition (eg. 0.6),
    # fs: sampling freq., nfilts: number of Bark subbands
    # nfilts: Number of subbands in the Bark domain, for instance 64
    # Returns: mTbark: the resulting Masking Threshold on the Bark scale

    # Compute the non-linear superposition:
    mTbark = np.dot(mXbark ** alpha, spreadingfuncmatrix ** alpha)
    # apply the inverse exponent to the result:
    mTbark = mTbark ** (1.0 / alpha)
    # Threshold in quiet:
    maxfreq = fs / 2.0
    maxbark = hz2bark(maxfreq)
    step_bark = maxbark / (nfilts - 1)
    barks = np.arange(0, nfilts) * step_bark
    # convert the bark subband frequencies to Hz:
    f = bark2hz(barks) + 1e-6
    # Threshold of quiet in the Bark subbands in dB:
    LTQ = np.clip((3.64 * (f / 1000.) ** -0.8 - 6.5 * np.exp(-0.6 * (f / 1000. - 3.3) ** 2.)
                   + 1e-3 * ((f / 1000.) ** 4.)), -20, 120)
    # Maximum of spreading functions and hearing threshold in quiet:
    mTbark = np.max((mTbark, 10.0 ** ((LTQ - 60) / 20)), 0)
    return mTbark


def hz2bark(f):
    """ Usage: Bark=hz2bark(f)
            f    : (ndarray)    Array containing frequencies in Hz.
        Returns  :
            Brk  : (ndarray)    Array containing Bark scaled values.
        """
    Brk = 6. * np.arcsinh(f / 600.)
    return Brk


def bark2hz(Brk):
    """ Usage: transfer bark size back to frequency
        Hz=bark2hs(Brk)
        Args     :
            Brk  : (ndarray)    Array containing Bark scaled values.
        Returns  :
            Fhz  : (ndarray)    Array containing frequencies in Hz.
        """
    Fhz = 600. * np.sinh(Brk / 6.)
    return Fhz


def mapping2barkmat(fs, nfilts, nfft):
    # Constructing mapping matrix W which has 1's for each Bark subband, and 0's else
    # usage: W=mapping2barkmat(fs, nfilts,nfft)
    # arguments: fs: sampling frequency
    # nfilts: number of subbands in Bark domain
    # nfft: number of subbands in fft
    maxbark = hz2bark(fs / 2)  # upper end of our Bark scale:22 Bark at 16 kHz
    nfreqs = nfft / 2;
    step_bark = maxbark / (nfilts - 1)
    binbark = hz2bark(np.linspace(0, (nfft / 2), int(nfft / 2) + 1) * fs / nfft)
    W = np.zeros((nfilts, nfft))
    for i in range(nfilts):
        W[i, 0:int(nfft / 2) + 1] = (np.round(binbark / step_bark) == i)
    return W


def mapping2bark(mX, W, nfft):
    # Maps (warps) magnitude spectrum vector mX from DFT to the Bark scale
    # arguments: mX: magnitude spectrum from fft
    # W: mapping matrix from function mapping2barkmat
    # nfft: : number of subbands in fft
    # returns: mXbark, magnitude mapped to the Bark scale
    nfreqs = int(nfft / 2)
    # Here is the actual mapping, suming up powers and conv. back to Voltages:
    mXbark = (np.dot(np.abs(mX[:nfreqs]) ** 2.0, W[:, :nfreqs].T)) ** (0.5)
    return mXbark


def mappingfrombarkmat(W, nfft):
    # Constructing inverse mapping matrix W_inv from matrix W for mapping back from bark scale
    # usuage: W_inv=mappingfrombarkmat(Wnfft)
    # argument: W: mapping matrix from function mapping2barkmat
    # nfft: : number of subbands in fft
    nfreqs = int(nfft / 2)
    W_inv = np.dot(np.diag((1.0 / (np.sum(W, 1) + 1e-6)) ** 0.5), W[:, 0:nfreqs + 1]).T
    return W_inv


# -------------------
def mappingfrombark(mTbark, W_inv, nfft):
    # usage: mT=mappingfrombark(mTbark,W_inv,nfft)
    # Maps (warps) magnitude spectrum vector mTbark in the Bark scale
    # back to the linear scale
    # arguments:
    # mTbark: masking threshold in the Bark domain
    # W_inv : inverse mapping matrix W_inv from matrix W for mapping back from bark scale
    # nfft: : number of subbands in fft
    # returns: mT, masking threshold in the linear scale
    # 从bark尺度转换回线性尺度
    nfreqs = int(nfft / 2)
    mT = np.dot(mTbark, W_inv[:, :nfreqs].T)
    return mT


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
    # audio_path = '/Users/yuxuanliu/PycharmProjects/Python-Audio-Coder/PythonPsychoacoustics/output/The_Latin_Fab_Four-Yesterday.wav'
    audio_path = '/Users/yuxuanliu/Downloads/wlh_nibuzhidaodeshi.wav'

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
    print("---------D1 of the mT_matrix----------", mT_matrix[:, 1])
    print("length", len(mT_matrix[:, 1]))

    plt.figure(figsize=(10, 4))
    print("changdu: ", mT_matrix.shape[1])
    librosa.display.specshow(librosa.amplitude_to_db(mT_matrix, ref=np.max),
                             sr=fs, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Masking Threshold over Time')
    plt.tight_layout()
    plt.show()



     # 以下代码为尝试生成最大阈值限制下的最大噪声
    
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



    '''
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
        noise_adjusted[:, i] = noise_magnitude * np.exp(1j * noise_phase)

    # 将调整后的噪声添加到原始信号的STFT
    S_noisy = S + noise_adjusted
    '''

    # 4. 逆STFT重建信号
    x_noisy = librosa.istft(S_noisy, hop_length=hop_length)
    x_only_noise = librosa.istft(noise_adjusted, hop_length=hop_length)

    # 5. 导出音频文件
    output_path = './output/noisy_nbzdds_whole.wav'
    sf.write(output_path, x_noisy, fs)
    # 仅仅输出噪声
    noise_output_path = './output/only_noise_nbzdds_whole.wav'
    sf.write(noise_output_path, x_only_noise, fs)
    print(f"Added noise audio file saved as {output_path}")

    # 计算信号的功率
    signal_power = np.mean(x ** 2)

    # 计算噪声的功率
    noise_power = np.mean(x_only_noise ** 2)

    # 计算信噪比
    SNR = 10 * np.log10(signal_power / noise_power)

    print(f"Signal-to-Noise Ratio (SNR): {SNR} dB")

    """
  # x=np.random.randn(32000)*1000 // TODO
  sound.sound(x,fs)

  mX=np.abs(np.fft.fft(x[0:2048],norm='ortho'))[0:1025]
  mXbark=mapping2bark(mX,W,nfft)
  #Compute the masking threshold in the Bark domain:
  mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts)
  #Massking threshold in the original frequency domain
  mT=mappingfrombark(mTbark,W_inv,nfft)
  plt.plot(20*np.log10(mX+1e-3))
  plt.plot(20*np.log10(mT+1e-3))
  plt.title('Masking Theshold for White Noise')
  plt.legend(('Magnitude Spectrum White Noise','Masking Threshold'))
  plt.xlabel('FFT subband')
  plt.ylabel("Magnitude ('dB')")
  plt.show()
  #----------------------------------------------
  #A test magnitude spectrum, an idealized tone in one subband:
  #tone at FFT band 200:
  x=np.sin(2*np.pi/nfft*200*np.arange(32000))*1000
  sound.sound(x,fs)

  mX=np.abs(np.fft.fft(x[0:2048],norm='ortho'))[0:1025]
  #Compute the masking threshold in the Bark domain:
  mXbark=mapping2bark(mX,W,nfft)
  mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts)
  mT=mappingfrombark(mTbark,W_inv,nfft)
  plt.plot(20*np.log10(mT+1e-3))
  plt.title('Masking Theshold for a Tone')
  plt.plot(20*np.log10(mX+1e-3))
  plt.legend(('Masking Trheshold', 'Magnitude Spectrum Tone'))
  plt.xlabel('FFT subband')
  plt.ylabel("dB")
  plt.show()

  #stft, norm='ortho':
  #import scipy.signal
  #f,t,y=scipy.signal.stft(x,fs=32000,nperseg=2048)
  #make it orthonormal for Parsevals Theorem:
  #Hann window power per sample: 0.375
  #y=y*sqrt(2048/2)/2/0.375
  #plot(y[:,1])
  #plot(mX)
  """

    """
  y=zeros((1025,3))
  y[0,0]=1
  t,x=scipy.signal.istft(y,window='boxcar')
  plot(x)
  #yields rectangle with amplitude 1/2, for orthonormality it would be sqrt(2/N) with overlap, 
  #hence we need a factor sqrt(2/N)*2 for the synthesis, and sqrt(N/2)/2 for the analysis
  #for othogonality.
  #Hence it needs factor sqrt(N/2)/2/windowpowerpersample, hence for Hann Window:
  #y=y*sqrt(2048/2)/2/0.375
  """

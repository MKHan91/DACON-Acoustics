# wave 파일 읽어오기
import librosa
import pandas as pd
import numpy as np
import os.path as osp
import os


from tqdm import tqdm

from scipy.signal import stft


def stft_to_mel(signal, sample_rate, min=None, max=None):
  """
  STFT 함수를 이용할때 중요한 인자가 있다.
  nperseg: 각 segment의 길이 (Window Size), 주파수 해상도는 주로 윈도우의 크기에 의해 결정됩니다.
           윈도우의 크기가 커질수록 더 많은 주파수 세분성을 얻을 수 있습니다.
           하지만 윈도우 크기를 증가시키면 시간 해상도가 저하될 수 있으므로 이를 고려해야 합니다.
           일반적으로는 주파수와 시간 해상도 간의 균형을 고려하여 적절한 윈도우 크기를 선택해야 합니다.
  noverlap: 오버랩은 인접한 윈도우 사이의 겹침 정도를 나타냅니다.
            오버랩을 증가시키면 각 윈도우 간의 시간 해상도가 향상되어 더 많은 주파수 정보를 얻을 수 있습니다.
            일반적으로 오버랩을 증가시키면 계산 비용이 더 많이 들지만 주파수 해상도가 향상됩니다.
  """
  f, t, Zxx = stft(signal, sample_rate, nperseg=512, noverlap=256)
  """
  f: Frequency axis (주파수 축). STFT의 주파수 영역을 나타낸다.
  t: Time axis (시간 축). 어떤 시간 지점에서 STFT가 계산되었는지 나타낸다.
  Zxx: STFT의 결과, 복소수 형태로 주파수 및 시간에 따른 신호의 주파수 성분을 나타냄.
  """
  Sxx = np.abs(Zxx) # Sxx: 주파수 영역에서 신호 세기를 나타냄. 즉, 주파수의 세기를 나타냄.
  Sxx = np.maximum(Sxx, 1e-8) # 주파수 세기의 최솟값을 1e-8로 제한하겠다.
  mel = 20*np.log10(Sxx) # 주파수의 세기를 dB로 표현.
  # mel = (mel + 160) / 160 # Normalization
  if min is None and max is None:
    return mel
  mel = (mel - min) / (max - min)   # Min-Max Scale Normalization
  """
  Mel은 사람의 청각에 대한 주파수 척도 나타냄.
  낮은 주파수(저주파)에서는 주파수 간격이 넓고 높은 주파수(고주파)에서는 주파수 간격이 좁아지는 것을 반영한다.
  """
  return mel


test_dir = osp.join(os.getcwd(), 'DATASET', 'test')
test_df_path = osp.join(os.getcwd(), 'DATASET', 'test.csv')

test_df = pd.read_csv(test_df_path)
test_filenames = test_df['SAMPLE_ID'].tolist()

signal_arr = []
outlier_signal = []
for num, file_name in tqdm(enumerate(test_filenames), total=len(test_filenames)):
    wave_path = osp.join(test_dir, file_name+'.wav')
    y, sr = librosa.load(wave_path, sr=None)

    if len(y) != 10 * sr:
        diff = len(signal_arr[0]) - len(y)
        y = np.concatenate([np.zeros(diff), y])
        # outlier_signal.append(y)
        signal_arr.append(y)
    else:
        signal_arr.append(y)

signal_arr = np.array(signal_arr)
signal_arr = np.reshape(signal_arr, (-1, sr*10))
np.save('./DATASET/test_dataset.npy', signal_arr)

test_dataset = np.load('./DATASET/test_dataset.npy')
sample_rates = np.array([int(data.shape[0]/10) for data in test_dataset])

mel_min = np.array([], dtype=np.float32)
mel_max = np.array([], dtype=np.float32)
for num, sample in enumerate(test_dataset):
    mel = stft_to_mel(sample, sample_rates[num])

    mel_min = np.concatenate([mel_min, [mel.min()]])
    mel_max = np.concatenate([mel_max, [mel.max()]])

stft_images = 0
for num, sample in enumerate(test_dataset):
    mel = stft_to_mel(sample, sample_rates[num], min=np.median(mel_min), max=np.median(mel_max))
    mel = mel[np.newaxis, :]
    if num == 0:
        stft_images = mel
        continue

    stft_images = np.concatenate([stft_images, mel], axis=0)

print(stft_images.shape)
np.save('./DATASET/stft_image_test_dataset.npy', stft_images)
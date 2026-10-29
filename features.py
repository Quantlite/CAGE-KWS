import numpy as np
import librosa

SR = 16000
FRAME_LENGTH = 400
FRAME_STRIDE = 160
N_FFT = 512
N_MELS = 40
CLIP_LEN = SR
N_FRAMES = 98


def log_mel(y: np.ndarray) -> np.ndarray:
    if len(y) < CLIP_LEN:
        y = np.pad(y, (0, CLIP_LEN - len(y)))
    else:
        y = y[:CLIP_LEN]
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=FRAME_STRIDE,
        win_length=FRAME_LENGTH, n_mels=N_MELS, power=2.0, center=False,
    )
    log_mel_spec = np.log(mel + 1e-6).astype(np.float32).T
    n = log_mel_spec.shape[0]
    if n < N_FRAMES:
        log_mel_spec = np.pad(log_mel_spec, ((0, N_FRAMES - n), (0, 0)))
    elif n > N_FRAMES:
        log_mel_spec = log_mel_spec[:N_FRAMES]
    return log_mel_spec


def feature_shape():
    return (N_FRAMES, N_MELS)

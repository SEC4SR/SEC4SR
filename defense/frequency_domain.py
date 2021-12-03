
import torch
import torch
import torchaudio
from scipy import signal
from torch_lfilter import lfilter

def DS(audio, param=0.5, fs=16000, same_size=True):
    
    assert torch.is_tensor(audio) == True
    
    down_ratio = param
    new_freq = int(fs * down_ratio)
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=new_freq, resampling_method='sinc_interpolation')
    up_sampler = torchaudio.transforms.Resample(orig_freq=new_freq, new_freq=fs, resampling_method='sinc_interpolation')
    down_audio = resampler(audio)
    new_audio = up_sampler(down_audio)
    if new_audio.shape != audio.shape and same_size: ## sometimes the returned audio may have longer size (usually 1 point)
        return new_audio[..., :audio.shape[1]]
    return new_audio

def LPF(new, fs=16000, wp=4000, param=8000, gpass=3, gstop=40, same_size=True):

    assert torch.is_tensor(new) == True

    ws = param
    wp = 2 * wp / fs
    ws = 2 * ws / fs
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
    b, a = signal.butter(N, Wn, btype='low', analog=False, output='ba')
    
    audio = new.T.to("cpu")
    a = torch.tensor(a, device="cpu", dtype=torch.float)
    b = torch.tensor(b, device="cpu", dtype=torch.float)
    new_audio = lfilter(b, a, audio).T
    return new_audio.to(new.device)

def BPF(new, fs=16000, wp=[300, 4000], param=[50, 5000], gpass=3, gstop=40, same_size=True):

    assert torch.is_tensor(new) == True

    ws = param
    wp = [2 * wp_ / fs for wp_ in wp]
    ws = [2 * ws_ / fs for ws_ in ws]
    N, Wn = signal.buttord(wp, ws, gpass, gstop, analog=False, fs=None)
    b, a = signal.butter(N, Wn, btype="bandpass", analog=False, output='ba', fs=None)

    audio = new.T.to("cpu")
    a = torch.tensor(a, device="cpu", dtype=torch.float)
    b = torch.tensor(b, device="cpu", dtype=torch.float)
    new_audio = lfilter(b, a, audio).T
    return new_audio.to(new.device)

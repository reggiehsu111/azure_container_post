from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.core.model import Model
import numpy as np
import keras
import librosa
from scipy import signal
import os

n_bands = 150
n_frames = 150
sample_rate = 22050


def read_audio(audio_path, target_fs=None, duration=4):
    (audio, fs) = librosa.load(audio_path, sr=None, duration=duration)
    # if this is not a mono sounds file
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def pad_trunc_seq_rewrite(x, max_len):
    if x.shape[1] < max_len:
        pad_shape = (x.shape[0], max_len - x.shape[1])
        pad = np.ones(pad_shape) * np.log(1e-8)
        x_new = np.hstack((x, pad))
    # no pad necessary - truncate
    else:
        x_new = x[:, 0:max_len]
    return x_new

def extract_features(sample_audio, bands, frames, file_ext="*.wav"):
    # 4 second clip with 50% window overlap with small offset to guarantee frames
    n_window = int(sample_rate * 4. / frames * 2) - 4 * 2
    # 50% overlap
    n_overlap = int(n_window / 2.)
    # Mel filter bank
    melW = librosa.filters.mel(sr=sample_rate, n_fft=n_window, n_mels=bands, fmin=0., fmax=8000.)
    # Hamming window
    ham_win = np.hamming(n_window)
    log_specgrams_list = []


    sound_clip, fn_fs = read_audio(sample_audio, target_fs=sample_rate)
    assert (int(fn_fs) == sample_rate)

    if sound_clip.shape[0] < n_window:
        print("File %s is shorter than window size - DISCARDING - look into making the window larger." % fn)

    # Skip corrupted wavs
    if sound_clip.shape[0] == 0:
        print("File %s is corrupted!" % fn)

    # Compute spectrogram
    [f, t, x] = signal.spectral.spectrogram(
        x=sound_clip,
        window=ham_win,
        nperseg=n_window,
        noverlap=n_overlap,
        detrend=False,
        return_onesided=True,
        mode='magnitude')
    x = np.dot(x.T, melW.T)
    x = np.log(x + 1e-8)
    x = x.astype(np.float32).T
    x = pad_trunc_seq_rewrite(x, frames)

    log_specgrams_list.append(x)

    log_specgrams = np.asarray(log_specgrams_list).reshape(len(log_specgrams_list), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    features = np.concatenate((features, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        # first order difference, computed over 9-step window
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        # for using 3 dimensional array to use ResNet and other frameworks
        features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 1])

    return np.array(features)  #, np.array(labels, dtype=np.int)


def init():
    global cnn
    model_path = Model.get_model_path(model_name = 'fannoise-predictor')
    keras.backend.clear_session()
    cnn = keras.models.load_model(model_path)


@rawhttp
def run(request):
    print("Request: [{0}]".format(request))
    
    
    if request.method == 'POST':
        file = request.files['file']
#         file.save('temp.wav')
#         reqBody = request.get_data(False)
#         # reqBody to sample_audio
#         sample_audio = reqBody
        
        features = extract_features(file, bands=n_bands, frames=n_frames)
        y_prob = cnn.predict(features, verbose=0)
        y_pred = np.argmax(y_prob, axis=-1)
        defect_code = {2: "Pass", 3: "Noise", 4: "Rpm", 5: "Vibration"}
        # For a real-world solution, you would load the data from reqBody
        # and send it to the model. Then return the response.
#         os.remove('temp.wav')

        # For demonstration purposes, this example just returns the posted data as the response.
        return AMLResponse(defect_code[int(y_pred)], 200)
    else:
        return AMLResponse("bad request", 500)

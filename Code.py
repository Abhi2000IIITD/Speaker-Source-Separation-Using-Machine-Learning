import os
import librosa
import numpy as np
from scipy.spatial.distance import euclidean
import soundfile as sf
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import log10

# Define directories
master_directory = r'C:\Users\abhip\Desktop\1st Sem\ML\Project\Dataset\SourceSeperation'
output_directory = r'C:\Users\abhip\Desktop\1st Sem\ML\Project\Generated_Audio'
noisy_audio_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Dataset_Training"
noise_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Noises"
output_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Cleaned_Audio"
os.makedirs(output_dir, exist_ok=True)
files = os.listdir(master_directory)
os.makedirs(output_directory, exist_ok=True)

# Step 1: Extract and Rename Files
def extract_and_rename(file):
    if file.startswith('id1'):
        label_part = file[2:].split('-')[0][1:]
    else:
        label_part = file.split('_')[0].replace('id', '')
    if 'gen' in file:
        suffix = '_4'
    elif '000' in file:
        suffix = '_3'
    elif len(file.split('_')) > 1:
        suffix = '_' + file.split('_')[1].split('.')[0]
    else:
        suffix = '_X'
    return label_part + suffix

def group_files_by_label(files):
    file_groups = {}
    for file in files:
        label = extract_and_rename(file)
        label_key = label.split('_')[0]
        if label_key not in file_groups:
            file_groups[label_key] = []
        file_groups[label_key].append((label, file))
    return file_groups

file_groups = group_files_by_label(files)

# Step 2: Load and Normalize Audio
def load_and_normalize(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    return y, sr, len(y)

audio_signals = {}
min_lengths = {}

for label, files in file_groups.items():
    audio_signals[label] = {}
    lengths = []
    for sub_label, file_name in files:
        file_path = os.path.join(master_directory, file_name)
        y, sr, length = load_and_normalize(file_path)
        audio_signals[label][sub_label] = (y, sr)
        lengths.append(length)
    min_lengths[label] = min(lengths)

for label, signals in audio_signals.items():
    min_length = min_lengths[label]
    for sub_label, (y, sr) in signals.items():
        audio_signals[label][sub_label] = (y[:min_length], sr)

# Step 3: Resample to Common Sampling Rate
def resample_to_common_rate(y, sr, target_sr):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

common_sr = 16000
for label, signals in audio_signals.items():
    for sub_label, (y, sr) in signals.items():
        audio_signals[label][sub_label] = resample_to_common_rate(y, sr, common_sr)

# Step 4: Signal Transformations
def sin_transform(signal): return np.sin(signal)
def cos_transform(signal): return np.cos(signal)
def exp_transform(signal): return np.exp(signal / np.max(np.abs(signal)))  # Normalized exp

transformed_signals = {}
for label, signals in audio_signals.items():
    transformed_signals[label] = {}
    for sub_label, (y, sr) in signals.items():
        transformed_signals[label][sub_label] = {
            'sin': sin_transform(y),
            'cos': cos_transform(y),
            'exp': exp_transform(y)
        }

# Step 5: Compute Cepstrum
def compute_cepstrum(y):
    spectrum = np.fft.fft(y)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)  # Add epsilon to avoid log(0)
    cepstrum = np.fft.ifft(log_spectrum).real
    return cepstrum

cepstra = {}
for label, sub_labels in transformed_signals.items():
    cepstra[label] = {}
    for sub_label, transform_data in sub_labels.items():
        cepstra[label][sub_label] = {key: compute_cepstrum(sig) for key, sig in transform_data.items()}

# Step 6: Compute Delta Cepstra for All Pairs
delta_cepstra = {}

for label, sub_labels in cepstra.items():
    delta_cepstra[label] = {}
    keys = list(sub_labels.keys())
    for i, key1 in enumerate(keys):
        for key2 in keys[i + 1:]:  # Avoid duplicate pairs
            if sub_labels[key1] and sub_labels[key2]:
                min_length = min(len(sub_labels[key1]['sin']), len(sub_labels[key2]['sin']))
                delta_key = f"delta_{key1.split('_')[1]}_{key2.split('_')[1]}"
                delta_cepstra[label][delta_key] = {
                    'sin': sub_labels[key1]['sin'][:min_length] - sub_labels[key2]['sin'][:min_length],
                    'cos': sub_labels[key1]['cos'][:min_length] - sub_labels[key2]['cos'][:min_length],
                    'exp': sub_labels[key1]['exp'][:min_length] - sub_labels[key2]['exp'][:min_length]
                }

# Step 7: Calculate Metrics
def calculate_psnr(clean_signal, noisy_signal):
    mse = np.mean((clean_signal - noisy_signal) ** 2)
    max_signal = np.max(np.abs(clean_signal))
    if mse == 0:  # Avoid division by zero
        return float('inf')  # Perfect match
    psnr = 20 * np.log10(max_signal / np.sqrt(mse))
    return psnr

def calculate_metrics(transform1, transform2):
    t1 = transform1.flatten()
    t2 = transform2.flatten()
    euclid_dist = euclidean(t1, t2)
    return euclid_dist

# Main calculation loop
metrics_results = {}

for label, deltas in delta_cepstra.items():
    metrics_results[label] = {}
    for delta_key, transforms in deltas.items():
        metrics_results[label][delta_key] = {}
        for transform_key, delta_signal in transforms.items():
            euclid = calculate_metrics(delta_signal, np.zeros_like(delta_signal))
            psnr = calculate_psnr(delta_signal, np.zeros_like(delta_signal))
            metrics_results[label][delta_key][transform_key] = {
                'psnr': psnr,
                'euclidean_distance': euclid
            }
            
def save_delta_audio(delta_cepstra, output_directory, sample_rate=16000):

    for label, deltas in delta_cepstra.items():
        for delta_key, transforms in deltas.items():
            for transform_type, signal in transforms.items():
                # Generate the filename
                clean_idx, noisy_idx = delta_key.split('_')[1], delta_key.split('_')[2]
                filename = f"{label}_delta_{clean_idx}_{noisy_idx}_{transform_type}.wav"
                filepath = os.path.join(output_directory, filename)

                # Normalize and save the audio
                normalized_signal = librosa.util.normalize(signal)
                sf.write(filepath, normalized_signal, sample_rate)

# Call the function to save all delta cepstra
save_delta_audio(delta_cepstra, output_directory, sample_rate=common_sr)
print("Done Noise extraction ")

#============================================================================================
# Transform Functions
def sin_transform(x):
    return np.sin(x)

def cos_transform(x):
    return np.cos(x)

def exp_transform(x):
    x_normalized = x / np.max(np.abs(x))
    return np.exp(x_normalized)

# Normalize amplitude
def normalize_amplitude(signal):
    return signal / np.max(np.abs(signal))

# Compute cepstrum
def compute_cepstrum(signal):
    spectrum = np.fft.fft(signal)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)  # Avoid log(0)
    cepstrum = np.fft.ifft(log_spectrum).real
    return cepstrum

# Map noisy audio to noise files
noisy_audio_files = [f for f in os.listdir(noisy_audio_dir) if f.endswith(".wav")]
noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]

mapping = {}
for noisy_file in noisy_audio_files:
    audio_id, source_index = noisy_file.split("_")
    source_index = source_index.split(".")[0]
    matching_noises = [
        f for f in noise_files if f.startswith(audio_id) and f"_{source_index}_" in f
    ]
    mapping[noisy_file] = matching_noises

# Process each noisy audio file
for noisy_file, noise_list in mapping.items():
    noisy_path = os.path.join(noisy_audio_dir, noisy_file)
    noisy_signal, sr = librosa.load(noisy_path, sr=None)
    noisy_signal = normalize_amplitude(noisy_signal)

    best_distance = float("inf")
    best_cleaned_audio = None

    for noise_file in noise_list:
        noise_path = os.path.join(noise_dir, noise_file)
        noise_signal, noise_sr = librosa.load(noise_path, sr=None)
        noise_signal = normalize_amplitude(noise_signal)

        # Resample to match sampling rates
        if sr != noise_sr:
            noise_signal = librosa.resample(noise_signal, orig_sr=noise_sr, target_sr=sr)

        # Trim signals to the same length
        min_len = min(len(noisy_signal), len(noise_signal))
        noisy_trimmed = noisy_signal[:min_len]
        noise_trimmed = noise_signal[:min_len]

        # Determine transform type
        if "_sin.wav" in noise_file:
            transform_func = sin_transform
            transform_name = "sin"
        elif "_cos.wav" in noise_file:
            transform_func = cos_transform
            transform_name = "cos"
        elif "_exp.wav" in noise_file:
            transform_func = exp_transform
            transform_name = "exp"
        else:
            print(f"Unknown transform for noise file: {noise_file}")
            continue

        # Apply transformations
        noisy_transformed = transform_func(noisy_trimmed)
        noise_transformed = transform_func(noise_trimmed)

        # Compute cepstra
        cepstrum_noisy = compute_cepstrum(noisy_transformed)
        cepstrum_noise = compute_cepstrum(noise_transformed)

        # Subtract cepstra
        delta_cepstrum = cepstrum_noisy - cepstrum_noise

        # Reconstruct signal
        log_spectrum_clean = np.fft.fft(delta_cepstrum)
        abs_spectrum_clean = np.exp(log_spectrum_clean)
        phase_noisy = np.angle(np.fft.fft(noisy_transformed))
        spectrum_clean = abs_spectrum_clean * np.exp(1j * phase_noisy)
        cleaned_signal = np.fft.ifft(spectrum_clean).real

        # Normalize cleaned signal
        cleaned_signal = normalize_amplitude(cleaned_signal)

        # Evaluate using Euclidean distance
        distance = euclidean(noisy_trimmed, cleaned_signal)

        # Save the best cleaned audio based on minimum distance
        if distance < best_distance:
            best_distance = distance
            best_cleaned_audio = cleaned_signal

    # Save the best cleaned audio
    if best_cleaned_audio is not None:
        output_filename = f"{os.path.splitext(noisy_file)[0]}_cleaned.wav"
        output_path = os.path.join(output_dir, output_filename)
        sf.write(output_path, best_cleaned_audio, sr)
        
print("Clean audio generated")
#=============================================================================================
train_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Dataset_Training"
cleaned_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Cleaned_Audio"
test_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Dataset_Testing"
output_dir = r"C:\Users\abhip\Desktop\1st Sem\ML\Project\Labelled Data\Cleaned_Testing_Audio"
os.makedirs(output_dir, exist_ok=True)

# Feature Extraction function
def extract_features(audio_path, sr=16000, n_mfcc=13, delta=True, delta_delta=True):
    try:
        y, _ = librosa.load(audio_path, sr=sr)
        if len(y) == 0:
            print(f"Warning: {audio_path} is empty.")
            return None

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Delta and Delta-Delta MFCCs
        if delta:
            mfcc_delta = librosa.feature.delta(mfccs)
            mfccs_mean = np.hstack([mfccs_mean, np.mean(mfcc_delta, axis=1)])
        
        if delta_delta:
            mfcc_delta2 = librosa.feature.delta(mfcc_delta)
            mfccs_mean = np.hstack([mfccs_mean, np.mean(mfcc_delta2, axis=1)])

        # Additional spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        rms_energy = np.mean(librosa.feature.rms(y=y))

        # More features
        harmonic_to_noise_ratio = np.mean(librosa.effects.harmonic(y))
        tempogram = np.mean(librosa.feature.tempogram(y=y, sr=sr), axis=1)

        # Combine all features
        features = np.hstack([mfccs_mean, spectral_centroid, spectral_bandwidth, spectral_rolloff,
                              zero_crossing_rate, chroma, spectral_flatness, rms_energy,
                              harmonic_to_noise_ratio, tempogram])

        return features
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None
# Prepare Training Data function
def prepare_training_data(train_dir, cleaned_dir):
    X, y = [], []
    for file in os.listdir(train_dir):
        if file.endswith(".wav"):
            train_path = os.path.join(train_dir, file)
            cleaned_file = file.replace(".wav", "_cleaned.wav")
            cleaned_path = os.path.join(cleaned_dir, cleaned_file)

            if os.path.exists(cleaned_path):
                noisy_features = extract_features(train_path)
                clean_features = extract_features(cleaned_path)

                if noisy_features is not None and clean_features is not None:
                    X.append(noisy_features)
                    y.append(clean_features)

    return np.array(X), np.array(y)

# Train SVM with Hyperparameter Tuning and Cross-validation
def train_svm_with_tuning(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.transform(y)

    param_grid = {
        'estimator__C': [0.1, 1, 10],
        'estimator__epsilon': [0.1, 0.2, 0.3],
        'estimator__gamma': [0.001, 0.01, 0.1]
    }

    model = MultiOutputRegressor(SVR())
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y_scaled)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")

    return grid_search.best_estimator_, scaler

# Function to calculate PSNR
def calculate_psnr(clean_features, predicted_features):
    mse = np.mean((clean_features - predicted_features) ** 2)
    psnr = 10 * log10(np.max(clean_features) ** 2 / mse)
    return psnr

# Function to calculate Euclidean Distance
def calculate_euclidean_distance(clean_features, predicted_features):
    return np.linalg.norm(clean_features - predicted_features)

# Predict and reconstruct audio
def predict_cleaned_audio(svm_model, test_dir, output_dir, scaler, sr=16000, n_mfcc=13):
    for file in os.listdir(test_dir):
        if file.endswith(".wav"):
            test_path = os.path.join(test_dir, file)
            features = extract_features(test_path)

            if features is None:
                print(f"Skipping {file}: Feature extraction failed.")
                continue

            try:
                # Predict clean features
                predicted_features = svm_model.predict([features])[0]
                # Calculate PSNR and Euclidean Distance
                psnr = calculate_psnr(features, predicted_features)
                euclidean_dist = calculate_euclidean_distance(features, predicted_features)
                print(f"PSNR for {file}: {psnr} dB")
                print(f"Euclidean Distance for {file}: {euclidean_dist}")

                # Generate multi-frame MFCC for reconstruction
                mfcc_frames = np.tile(predicted_features, (20, 1)).T  # Replicate features over 20 frames

                # Reconstruct clean audio
                S = librosa.feature.inverse.mfcc_to_audio(
                    mfcc_frames,
                    sr=sr,
                    n_iter=32
                )
                reconstructed_audio = S / np.max(np.abs(S)) if np.max(np.abs(S)) != 0 else S
                cleaned_path = os.path.join(output_dir, file.replace(".wav", "_cleaned.wav"))
                sf.write(cleaned_path, reconstructed_audio, sr)
            except Exception as e:
                print(f"Failed to reconstruct audio for {file}: {e}")

# Main script
if __name__ == "__main__":
    # Prepare training data
    X, y = prepare_training_data(train_dir, cleaned_dir)
    print(f"Training data prepared: X.shape={X.shape}, y.shape={y.shape}")

    # Train model
    model, scaler = train_svm_with_tuning(X, y)

    # Predict and reconstruct audio files
    predict_cleaned_audio(model, test_dir, output_dir, scaler)
#==========================================================================================
# Speaker-Source Separation Using Machine Learning

This project focuses on developing a robust method for speaker-source separation from noisy audio recordings. By leveraging signal processing techniques and machine learning models, the aim is to effectively separate clean speech from noise and restore the original clean audio.

---

## 🚀 **Project Overview**
The goal of this project is to:
- Analyze noise characteristics across multiple versions of the same audio sample.
- Apply signal processing techniques like cepstral analysis, transformations (sine, cosine, exponential), and Fourier Transforms (FFT/IFFT).
- Subtract noise components to isolate clean audio.
- Evaluate the results using metrics like **Euclidean Distance** and **PSNR (Peak Signal-to-Noise Ratio)**.

---

## 📂 **Dataset Description**
The dataset consists of audio samples where each unique sample is associated with four noisy versions. Each file is labeled using a unique identifier.

## 🔢 **Mathematical Approach**

### 1. **Signal Transformation**
- Sine Transformation
- Cosine Transformation
- Exponential Transformation

### 2. **Cepstrum Computation**
- **FFT of Signal:**  
  X(f) = FFT(y(t))  
  where y(t) is the time-domain signal and X(f) is its frequency-domain representation.

- **Logarithmic Spectrum:**  
  log(X(f)) = log(|X(f)| + ε)  
  Here, ε is a small constant added to avoid taking the logarithm of zero.

- **Inverse FFT to Get Cepstrum:**  
  e(t) = IFFT(log(X(f)))  
  where e(t) is the Cepstrum (inverse FFT of the log-spectrum).

## 🛠️ **Methodology**
1. **Resampling:** All audio signals are resampled to 16 kHz for consistency.
2. **Signal Transformations:** Apply sine, cosine, and exponential transformations to the audio signals.
3. **Cepstrum Computation:** 
   - Perform FFT on the signal.
   - Apply logarithmic scaling to the spectrum.
   - Use IFFT to compute the Cepstrum.
4. **Delta Cepstral Computation:** Calculate the difference between the Cepstra of noisy and noise signals.
5. **Noise Subtraction:** Subtract the noise Cepstrum from the noisy signal’s Cepstrum to isolate clean audio.

---

## 🎓 **Model Training**
### 1. **Feature Extraction**
Extract audio features, including:
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Delta-MFCCs
- Spectral Centroid
- Zero-Crossing Rate
- Chroma Features
- RMS (Root Mean Square)

### 2. **Training Data Preparation**
Prepare the training data by:
- Reading noisy and clean audio files.
- Extracting features.
- Aligning them into feature vectors (X: Input for noisy signals, y: Target for clean signals).

### 3. **Model Selection**
The **Support Vector Machine (SVM)** was chosen for its effectiveness in handling high-dimensional feature spaces and robustness in small datasets.

---

## 📊 **Analysis**
### 1. **Models Considered**
- **Support Vector Machine (SVM):** Best suited for small datasets with high-dimensional feature spaces.
- **K-Nearest Neighbors (KNN):** Less effective for small datasets due to its reliance on pairwise distance calculations.
- **Gaussian Mixture Models (GMM):** May overfit with limited data.

### 2. **Performance**
- SVM with non-linear kernels (e.g., RBF) provided a good balance between flexibility and complexity.
- SVM outperformed KNN and GMM in avoiding overfitting.

---

## 📈 **Results**
- **PSNR (Peak Signal-to-Noise Ratio):** Achieved an average value of **25.0**.
- **Euclidean Distance:** Achieved a range of **3000–5000** across test samples.

---

## 📦 **Getting Started**
### Prerequisites
- Python 3.8+
- Libraries: `numpy`, `scipy`, `librosa`, `sklearn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abhi2000IIITD/Speaker-Source-Separation-Using-Machine-Learning.git
   cd Speaker-Source-Separation-Using-Machine-Learning

## ⚠️ Disclaimer

**Note:** The content in this repository is provided for educational and research purposes only. Any misuse, including direct copying leading to plagiarism, is the sole responsibility of the user. The owner is not liable for any plagiarism cases reported by academic or professional authorities.  

Users are encouraged to use this work ethically, giving proper credit when applicable. By using this repository, you agree to these terms.


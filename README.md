## Deepfake Audio Detection using Mel-Spectrograms

# About the project

This project is about detecting whether an audio clip is real or artificially generated (deepfake). With voice cloning and AI-generated speech becoming more advanced, it’s getting harder to trust audio content online.

So we built a deep learning-based system that can analyze audio and classify it as genuine or fake using Mel-spectrogram representations.

Instead of working directly with raw audio signals, we convert them into a format that makes patterns easier for a model to understand.

# Basic idea

The main idea is actually pretty straightforward:

Take an audio file
Convert it into a Mel-spectrogram (audio → visual representation)
Treat it like an image
Use a CNN model to classify it as real or fake

So basically:

Audio signal → Spectrogram image → CNN → Prediction

# Why Mel-Spectrogram?

We used Mel-spectrograms because:

Raw audio is hard for models to interpret directly
Mel-spectrograms show frequency vs time patterns
They are closer to how humans perceive sound
CNNs work really well on image-like data

So instead of forcing the model to “listen”, we let it “see the sound”.
# Pipeline
Audio File
     │
     ▼
Audio Loading (Librosa)
     │
     ▼
Mel-Spectrogram Generation
     │
     ▼
Log Scaling & Normalization
     │
     ▼
Resize to 224 × 224
     │
     ▼
Convert to 3 Channels
     │
     ▼
EfficientNet-B0
     │
     ▼
Classification
(Real / Fake)

# Methodology
1. Data preprocessing
Audio files were loaded using standard audio processing libraries
All audio clips were resampled to a fixed sampling rate for consistency
Silence trimming and normalization were applied where needed
2. Feature extraction
Each audio file was converted into a Mel-spectrogram(time vs frequency graph with different colors with varying intensity)
This transforms the waveform into a 2D matrix (time × frequency), It has varying dimensions accordin to the audio file so we resize it to 224 x 224.
The spectrogram is then treated like an image input for the CNN
3. Model architecture
I chose EffiecientNet, this expects the input in RGB, so we make it 3 channels.
EfficientNet is a family of CNN architectures designed to achieve high accuracy while maintaining computational efficiency.

Why EfficientNet-B0?
Lightweight architecture
Faster training compared to larger CNNs
Excellent transfer learning performance
Strong feature extraction capabilities
Transfer Learning

Instead of training from scratch, pretrained ImageNet weights were used.

# Feature Extraction

Most convolutional layers were frozen.

Only:

Classification layer
Final feature blocks

were trained.

Benefits:

Faster convergence
Reduced overfitting
Lower computational requirements

The model generally includes:

Convolution layers (for feature extraction)
Pooling layers (to reduce dimensionality)
Fully connected layers (for classification)
Dropout layers (to reduce overfitting)
4. Training setup
Supervised learning approach (real vs fake labels)
Loss function: Cross-entropy loss
Optimizer: Adam
Train-test split for evaluation
Trained for multiple epochs until convergence

# Evaluation Metrics

Several metrics were used to evaluate model performance.

1. Accuracy:
Measures the percentage of correctly classified samples.

Formula:
Accuracy=TP+TN+FP+FN/TP+TN
Result:Accuracy = 84.95%

2. Precision:
Measures how many samples predicted as fake/real are actually correct.

Precision=TP+FP/TP
Result: Precision = 77.75%

3. Recall:
Measures how many actual positive samples are correctly detected.

Recall=TP+FN/TP
Result : Recall = 96.93%

This is particularly important because missing a deepfake is generally more harmful than incorrectly flagging a genuine sample.

4. F1 Score: The harmonic mean of Precision and Recall.
F1=2⋅Precision+Recall/Precision⋅Recall
Result : F1 Score = 86.29%

5. Equal Error Rate (EER):
EER is widely used in verification and biometric systems.

It represents the point where: False Acceptance Rate = False Rejection Rate
Result : EER = 0.0985

Lower values indicate better model performance.

6. Confusion Matrix
                Predicted
              REAL   FAKE

Actual REAL    752    271
Actual FAKE     30    947
Observations
Only 30 fake samples were missed.
The model successfully detected most deepfake audio samples.
Some genuine audio clips were incorrectly classified as fake, resulting in lower precision.

We prioritized recall because in deepfake detection, missing a fake audio is more critical than misclassifying a real one.

# What we observed
Spectrogram-based features work surprisingly well for audio classification
CNNs can still perform strongly on audio tasks when data is represented properly
Deepfake detection is challenging because fake audio keeps improving over time
Model performance depends heavily on dataset quality and diversity

# Tech stack
Python
Librosa (audio processing)
NumPy, Pandas
Matplotlib (for visualization)
PyTorch / TensorFlow (CNN model)


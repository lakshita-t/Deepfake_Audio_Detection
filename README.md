Deepfake Audio Detection using Mel-Spectrograms

About the project

This project is about detecting whether an audio clip is real or artificially generated (deepfake). With voice cloning and AI-generated speech becoming more advanced, it’s getting harder to trust audio content online.

So we built a deep learning-based system that can analyze audio and classify it as genuine or fake using Mel-spectrogram representations.

Instead of working directly with raw audio signals, we convert them into a format that makes patterns easier for a model to understand.

Basic idea

The main idea is actually pretty straightforward:

Take an audio file
Convert it into a Mel-spectrogram (audio → visual representation)
Treat it like an image
Use a CNN model to classify it as real or fake

So basically:

Audio signal → Spectrogram image → CNN → Prediction

  Why Mel-Spectrogram?

We used Mel-spectrograms because:

Raw audio is hard for models to interpret directly
Mel-spectrograms show frequency vs time patterns
They are closer to how humans perceive sound
CNNs work really well on image-like data

So instead of forcing the model to “listen”, we let it “see the sound”.

  Methodology
1. Data preprocessing
Audio files were loaded using standard audio processing libraries
All audio clips were resampled to a fixed sampling rate for consistency
Silence trimming and normalization were applied where needed
2. Feature extraction
Each audio file was converted into a Mel-spectrogram
This transforms the waveform into a 2D matrix (time × frequency)
The spectrogram is then treated like an image input for the CNN
3. Model architecture

We used a Convolutional Neural Network (CNN) because:

It can detect spatial patterns in spectrograms
It works well for image-like representations

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
  Results

The model performed fairly well considering limited data and training time:

Accuracy: ~86%
Recall: High (good at detecting fake audio)

We prioritized recall because in deepfake detection, missing a fake audio is more critical than misclassifying a real one.

  What we observed
Spectrogram-based features work surprisingly well for audio classification
CNNs can still perform strongly on audio tasks when data is represented properly
Deepfake detection is challenging because fake audio keeps improving over time
Model performance depends heavily on dataset quality and diversity
  Tech stack
Python
Librosa (audio processing)
NumPy, Pandas
Matplotlib (for visualization)
PyTorch / TensorFlow (CNN model)


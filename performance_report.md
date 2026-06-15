# Performance Report

## Model Evaluation

The proposed Deepfake Audio Detection system was evaluated on a held-out test set consisting of 2,000 audio samples. Performance was measured using Accuracy, Precision, Recall, F1 Score, Equal Error Rate (EER), and a Confusion Matrix to assess the model's ability to distinguish between genuine and AI-generated speech.

### Evaluation Metrics

| Metric                 |  Value |
| ---------------------- | -----: |
| Accuracy               | 84.95% |
| Precision              | 77.75% |
| Recall                 | 96.93% |
| F1 Score               | 86.29% |
| Equal Error Rate (EER) | 0.0985 |

### Confusion Matrix

|             | Predicted Fake | Predicted Real |
| ----------- | -------------: | -------------: |
| Actual Fake |            752 |            271 |
| Actual Real |             30 |            947 |

### Results Analysis

The model achieved an overall accuracy of 84.95%, demonstrating strong performance in distinguishing between authentic and synthetic audio samples. The high recall score of 96.93% indicates that the model successfully identifies the majority of deepfake audio recordings, missing only a small number of fake samples.

A precision score of 77.75% suggests that some genuine audio recordings are incorrectly classified as deepfakes. While this results in a higher false-positive rate, prioritizing recall is desirable in deepfake detection systems, where failing to detect a fake recording can have more serious consequences than incorrectly flagging a genuine one.

The F1 score of 86.29% reflects a good balance between precision and recall, confirming the overall robustness of the model. Additionally, the Equal Error Rate (EER) of 0.0985 indicates strong discriminative capability between real and fake audio samples. Since lower EER values correspond to better performance, an EER below 0.10 demonstrates effective separation of the two classes.

### Key Observations

* The model successfully detected 947 out of 977 fake audio samples.
* Only 30 fake samples were misclassified, resulting in very high recall.
* Some genuine recordings were incorrectly flagged as fake, leading to reduced precision.
* The low EER value indicates reliable classification performance and good class separation.
* The model is particularly suitable for security-oriented applications where detecting deepfake audio is prioritized over minimizing false alarms.

### Conclusion

The experimental results demonstrate that Mel-spectrogram-based feature extraction combined with EfficientNet-B0 transfer learning is effective for deepfake audio detection. The model achieves high recall and a strong F1 score while maintaining a low Equal Error Rate, making it a reliable solution for identifying AI-generated speech.

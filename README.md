# ASL-Alphabet Image Classification

This project contains approaches to classify letter/alphabet images that contain gestures of the American Sign Language (ASL). Deep Learning models are used with Keras, including CNNs defined from scratch, ResNet50 with fine tuning and autoencoders in combination with random forests.

The original dataset can be downloaded from Kaggle: [asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

![ASL Examples](./assets/asl_examples_labelled.jpg)

![Autoencoder + Random Forest: Confusion Matrix](./assets/confusion_matrix_autoencoder_rf.jpeg)

![From Scratch CNN: Confusion Matrix](./assets/confusion_matrix_scratch.jpg)

![Transfer Learning: Confusion Matrix](./assets/confusion_matrix_transfer_learning.jpg)

![Autoencoder: Reconstruction](./assets/autoencoder.jpg)

![Autoencoder: Missclassifications](./assets/autoencoder_missclassifications.jpg)

![From Scratch CNN: Missclassifications](./assets/cnn_scratch_missclassifications.jpg)

![From Scratch CNN: Learning Curves](./assets/cnn_scratch_learning_curves.jpg)





## Authorship

Mikel Sagardia, 2022.  
No guarantees.

### Requirements

- Submit report as a PDF.
- Deep learning model for any task we select, using the dataset of our choice.
- Describe the dataset.
- Explain main objectives: problem type, goals.
- Describe briefly: data exploration, cleaning, feature engineering.
- Variations of a deep learning model: at least 3; different hyperparameters, etc.
- Explain model recommendation: choose explainability / accuracy?
- Explain key findings.
- Next steps: issues, improvements, etc.
# Exploiting the Potential of Unlabeled Data: a Reliable Semi-Supervised Framework for Speech Emotion Recognition

- This GitHub repo contains the code of Submission #1954 in Interspeech 2024.

- Structure of the repo is as follows:
```
├── augmentation                # Files of augmentation for speech data
│   ├── augmentation.py     # Implementation of all augmentation methods

└── fixmatch                # Files of FixMatch implementation with Wav2Vec 2.0
│   ├── model.py            # Implementation of the FixMatch model
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code

└── npl                     # Files of FixMatch implementation with Negative Pseudo-labeling
│   ├── model.py            # Implementation of the FixMatch + NPL model
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
│   └── loss.py             # NPL loss function

└── proposed                # Files of our proposed model
│   ├── model.py            # Implementation of the FixMatch + NPL + CC model
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
│   └── loss.py             # NPL and Confidence Calibration loss functions

└── wav2vec                 # Files of Wav2Vec 2.0
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
```
- Requirements to run the code:
```
scikit-learn
numpy
torch
torchaudio
transformers
librosa
soundfile
speechaugs
```

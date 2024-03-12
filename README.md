# Unleashing the Potential of Unlabeled Speech: A Trustworthy Pseudo-Labeling Framework for Semi-Supervised Emotion Recognition

- This GitHub repo contains the code of Submission #1954 in Interspeech 2024.

- Structure of the repo is as follows:
```
├── fixmatch                # Files of FixMatch implementation with Wav2Vec 2.0
│   ├── model.py            # Implementation of the FixMatch model
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
└── npl                     # Files of FixMatch implementation with Negative Pseudo-labeling
│   ├── model.py            # Implementation of the FixMatch + NPL model
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
└── proposed                # Files of our proposed model
│   ├── model.py            # Implementation of the FixMatch + NPL + CC model
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
└── wav2vec                 # Files of Wav2Vec 2.0
│   ├── train.py            # Training code
│   └── eval.py             # Evaluation code
```

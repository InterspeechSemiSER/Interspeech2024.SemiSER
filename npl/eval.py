from multiprocessing import get_context
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torch
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from model import NPL
import warnings
import numpy as np
from confidence_intervals import evaluate_with_conf_int
from confidence_intervals.utils import create_data


warnings.filterwarnings('ignore')

device = 'cuda'

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
target_names = ['hap', 'neu', 'sad', 'ang']

def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["input_length"] = len(batch["speech"]) / sampling_rate
    batch["label"] = batch["emotion_id"]
    return batch

def map_to_pred(batch, pool, model):
    input_values = [b for b in batch["speech"]]
    inputs = processor(input_values, sampling_rate=16_000, padding=True, return_tensors="pt").to(device)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.nn.functional.softmax(logits, dim=-1)   
    
    scores, _ = torch.max(scores, 1)
    batch['confidence_score'] = [float(i) for i in scores]

    predicted_class_ids = torch.argmax(logits, dim=-1)
    batch['prediction'] = [int(i) for i in predicted_class_ids]

    batch.pop('speech')
    batch.pop('sampling_rate')
    batch.pop('input_length')
    return batch

def compute_f1(preds, labels):
    return f1_score(labels, preds, average='weighted')

def inference(data_path, checkpoint_path, save_path='tmp.csv'):
    test_dataset = load_dataset(
        "csv", data_files=data_path, split="train", cache_dir='path to cache dir')

    test_dataset = test_dataset.map(
        speech_file_to_array_fn, num_proc=1)

    model = NPL.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    with get_context("fork").Pool(processes=2) as pool:
        result = test_dataset.map(
            map_to_pred, batched=True, batch_size=2, fn_kwargs={"pool": pool, "model": model}, load_from_cache_file=False
        )

    result.set_format(type="pandas", columns=[
                            "path", "prediction", "label", "confidence_score"])

    result.to_csv(save_path)
    data = pd.read_csv(save_path)
    
    prediction = np.array(data['prediction'])
    label = np.array(data['label'])
    
    print('Acc', evaluate_with_conf_int(prediction, accuracy_score, label, None, num_bootstraps=1000, alpha=5))
    print('W.Acc', evaluate_with_conf_int(prediction, balanced_accuracy_score, label, None, num_bootstraps=1000, alpha=5))
    print('F1', evaluate_with_conf_int(prediction, compute_f1, label, None, num_bootstraps=1000, alpha=5))

inference('path to test.csv', 'path to checkpoint')

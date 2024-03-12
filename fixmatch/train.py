import numpy as np
from datasets import load_dataset, load_metric

import os

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2FeatureExtractor
from model import FixMatch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
import torchaudio
from ..augmentation.augmentation import *

checkpoint_dir = "path to save checkpoints"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

dataset = load_dataset('csv', data_files='path to dataset', split='train', cache_dir='path to save cache')

# Filter the dataset based on the 'split' column
train_dataset = dataset.filter(lambda example: example['split'] == 'train')
valid_dataset = dataset.filter(lambda example: example['split'] == 'valid')
test_dataset = dataset.filter(lambda example: example['split'] == 'test')

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    padding_side="right",
    do_normalize=True,
    return_attention_mask=False,
)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    return batch


train_dataset = train_dataset.map(speech_file_to_array_fn)
test_dataset = test_dataset.map(speech_file_to_array_fn)


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {feature_extractor.sampling_rate}."

    batch["input_values"] = feature_extractor(
        batch["speech"], sampling_rate=batch["sampling_rate"][0]
    ).input_values

    batch["label"] = batch["emotion_id"]
    batch["is_groundtruth"] = batch["is_groundtruth"]
    return batch


train_dataset = train_dataset.map(
    prepare_dataset, batch_size=8, num_proc=4, batched=True
)
test_dataset = test_dataset.map(prepare_dataset, batch_size=8, num_proc=4, batched=True)


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_values = self.processor.pad(
            [{"input_values": feature["input_values"]} for feature in features],
            return_tensors="pt",
            padding=True,
        )
        weak_augmented_input_values = self.processor.pad(
            [
                {"input_values": weak_augment_with_time_mask(feature["input_values"])}
                for feature in features
            ],
            return_tensors="pt",
            padding=True,
        )
        strong_augmented_input_values_1 = self.processor.pad(
            [
                {"input_values": strong_augment_with_noise(feature["input_values"])}
                for feature in features
            ],
            return_tensors="pt",
            padding=True,
        )
        strong_augmented_input_values_2 = self.processor.pad(
            [
                {
                    "input_values": strong_augment_with_speed_perturbation(
                        feature["input_values"]
                    )
                }
                for feature in features
            ],
            return_tensors="pt",
            padding=True,
        )

        labels = torch.tensor([feature["label"] for feature in features])
        is_groundtruth = torch.tensor([feature["is_groundtruth"] for feature in features])

        return {
            "input_values": input_values["input_values"],
            "weak_augmented_input_values": weak_augmented_input_values["input_values"],
            "strong_augmented_input_values_1": strong_augmented_input_values_1[
                "input_values"
            ],
            "strong_augmented_input_values_2": strong_augmented_input_values_2[
                "input_values"
            ],
            "labels": labels,
            "is_groundtruth": is_groundtruth,
            
        }


data_collator = DataCollatorCTCWithPadding(processor=feature_extractor, padding=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {"acc": accuracy_score(labels, predictions)}


model = FixMatch.from_pretrained(
    'facebook/wav2vec2-base',
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    final_dropout=0.1,
    mask_time_prob=0.05,
    layerdrop=0.1,
    num_labels=4,
    classifier_proj_size=256,
    use_weak_augment=False,
    use_pseudo_label=True,
    cache_dir='path to save cache'
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir=checkpoint_dir,
    group_by_length=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=20,
    eval_accumulation_steps=1,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    dataloader_num_workers=6,
    learning_rate=1.5e-5,
    warmup_steps=0,
    save_total_limit=20,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# trainer.train()


trainer.train(resume_from_checkpoint=False)

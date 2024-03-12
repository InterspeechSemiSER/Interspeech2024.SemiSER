import torch.nn as nn
import torch
from transformers import (
    HubertModel,
    Wav2Vec2PreTrainedModel,
    HubertPreTrainedModel,
    Wav2Vec2Config,
    HubertConfig,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2FeatureEncoder,
    Wav2Vec2FeatureProjection,
)
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

import math
from typing import Any, Dict, List, Optional, Union
import torchaudio
import os
import random

from loss import npl_loss

_HIDDEN_STATES_START_POSITION = 2
pretrain_processor_wav2vec2 = "facebook/wav2vec2-base-100h"
pretrain_audio_model_wav2vec2 = "facebook/wav2vec2-base"


class NPL(Wav2Vec2PreTrainedModel):
    def __init__(self, config, use_weak_augment=True, use_pseudo_label=False):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        self.encoder_dropout = nn.Dropout1d(0.3)
        self.groundtruth_confidence_threshold = 0.85
        self.pseudo_label_confidence_threshold = 0.95

        self.use_weak_augment = use_weak_augment
        print("Using weak augment:", self.use_weak_augment)
        self.post_init()

    def freeze_feature_extractor(self):
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward_encoder(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            True if self.config.use_weighted_layer_sum else output_hidden_states
        )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        return hidden_states, outputs

    def forward_decoder(self, hidden_states, attention_mask=None):
        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1], attention_mask
            )
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(
                -1, 1
            )

        logits = self.classifier(pooled_output)
        return logits

    def forward_fp(self, hidden_states, attention_mask=None):
        hidden_states_fp = self.encoder_dropout(hidden_states)
        assert hidden_states_fp.shape == hidden_states.shape
        return self.forward_decoder(hidden_states_fp, attention_mask=attention_mask)

    def split_label_unlabeled(self, logits, labels, is_groundtruth):
        labeled_count = (is_groundtruth == 1).sum().item()
        unlabeled_count = (is_groundtruth == 0).sum().item()

        logits_groundtruth = logits[is_groundtruth == 1][:labeled_count]
        logits_pseudo_label = logits[is_groundtruth == 0][:unlabeled_count]

        labels_groundtruth = labels[is_groundtruth == 1][:labeled_count]
        labels_pseudo_label = labels[is_groundtruth == 0][:unlabeled_count]

        # check if groundtruth plus pseudo label is equal to logits
        assert (
            logits_groundtruth.shape[0] + logits_pseudo_label.shape[0]
            == logits.shape[0]
        )

        return (
            logits_groundtruth,
            logits_pseudo_label,
            labels_groundtruth,
            labels_pseudo_label,
        )

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        weak_augmented_input_values: Optional[torch.Tensor] = None,
        strong_augmented_input_values_1: Optional[torch.Tensor] = None,
        strong_augmented_input_values_2: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        is_groundtruth: Optional[torch.Tensor] = None,
    ):
        # inference mode
        if input_values is not None and labels is None:
            hidden_states, outputs = self.forward_encoder(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = self.forward_decoder(hidden_states, attention_mask=attention_mask)
            return SequenceClassifierOutput(
                loss=None,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        # training mode
        else:
            if self.use_weak_augment:
                hidden_states_w, outputs_w = self.forward_encoder(
                    weak_augmented_input_values,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                hidden_states_w, outputs_w = self.forward_encoder(
                    input_values,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            hidden_states_s, _ = self.forward_encoder(
                strong_augmented_input_values_1,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # weak augmented output
            logits_w = self.forward_decoder(
                hidden_states_w, attention_mask=attention_mask
            )

            # strong augmented output with speed perturbation
            logits_s = self.forward_decoder(
                hidden_states_s, attention_mask=attention_mask
            )

            (
                logits_w_groundtruth,
                logits_w_pseudo_label,
                labels_groundtruth,
                labels_pseudo_label,
            ) = self.split_label_unlabeled(logits_w, labels, is_groundtruth)
            
            logits_s_groundtruth, logits_s_pseudo_label, _, _ = (
                self.split_label_unlabeled(logits_s, labels, is_groundtruth)
            )
            
            loss = None

            # if groundtruth are there, calculate groundtruth loss
            if logits_w_groundtruth.shape[0] > 0:
                loss_w = CrossEntropyLoss()(
                    logits_w_groundtruth.view(-1, self.config.num_labels),
                    labels_groundtruth.view(-1),
                )
                # if pseudo label are there, calculate pseudo label loss
                if logits_w_pseudo_label.shape[0] > 0:
                    preds_w = torch.argmax(logits_w_pseudo_label, dim=-1).to(
                        self.device
                    )
                    confidence_w = torch.softmax(logits_w_pseudo_label, dim=-1).to(
                        self.device
                    )

                    # only compute loss where pseudo label with confidence > threshold
                    mask = (
                        confidence_w.max(dim=-1).values
                        > self.pseudo_label_confidence_threshold
                    )
                    
                    loss_s = CrossEntropyLoss()(
                        logits_s_pseudo_label[mask].view(-1, self.config.num_labels),
                        preds_w[mask].view(-1),
                    )
                    
                    loss_npl = npl_loss(logits_w_pseudo_label, logits_s_pseudo_label, 1, self.config.num_labels)
                
                    loss_s = loss_s + loss_npl
                # if no pseudo label, use groundtruth preds to calculate loss
                else:
                    preds_w = torch.argmax(logits_w_groundtruth, dim=-1).to(self.device)
                    confidence_w = torch.softmax(logits_w_groundtruth, dim=-1).to(
                        self.device
                    )

                    # only compute loss where pseudo label with confidence > threshold
                    mask = (
                        confidence_w.max(dim=-1).values
                        > self.pseudo_label_confidence_threshold
                    )
                    
                    loss_s = CrossEntropyLoss()(
                        logits_s_groundtruth[mask].view(-1, self.config.num_labels),
                        preds_w[mask].view(-1),
                    )
                    
                    loss_npl = 0.0
                
                    loss_s = loss_s + loss_npl
                    
            # if no groundtruth, use pseudo label to calculate loss
            else:
                loss_w = 0.0
                preds_w = torch.argmax(logits_w_pseudo_label, dim=-1).to(self.device)
                confidence_w = torch.softmax(logits_w_pseudo_label, dim=-1).to(
                    self.device
                )

                # only compute loss where pseudo label with confidence > threshold
                mask = (
                    confidence_w.max(dim=-1).values
                    > self.pseudo_label_confidence_threshold
                )

                loss_s = CrossEntropyLoss()(
                    logits_s_pseudo_label[mask].view(-1, self.config.num_labels),
                    preds_w[mask].view(-1),
                )
                
                loss_npl = npl_loss(logits_w_pseudo_label, logits_s_pseudo_label, 1, self.config.num_labels)
                
                loss_s = loss_s + loss_npl

            loss = (loss_w + loss_s) / 2.0

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits_w,
                hidden_states=outputs_w.hidden_states,
                attentions=outputs_w.attentions,
            )

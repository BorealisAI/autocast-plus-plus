# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Autocast (https://arxiv.org/abs/2206.15474) implementation
# from https://github.com/andyzoujm/autocast by Andy Zou and Tristan Xiao and Ryan Jia and Joe Kwon and Mantas Mazeika and Richard Li and Dawn Song and Jacob Steinhardt and Owain Evans and Dan Hendrycks
####################################################################################


import torch
import copy
import transformers
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput


class FiDT5(transformers.T5ForConditionalGeneration):
    """
    Modified Fusion-in-Decoder T5 model.
    """

    def __init__(self, config, logger):
        super().__init__(config)
        self.logger = logger
        self.wrap_encoder()

        # attention for alignment loss
        tf_enc = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=8,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout_rate,
            activation="relu",
            batch_first=True,
        )
        self.aln_tf = nn.TransformerEncoder(tf_enc, num_layers=2)
        self.aln_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, 1),
        )

        # count number of trinable parameter per componenet
        component_n_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                name_prefix = name.split(".")[0]
                if name_prefix not in component_n_params:
                    component_n_params[name_prefix] = 0
                component_n_params[name_prefix] += param.numel()
        self.logger.info("Breakdown of number of trainable parameters per component:")
        for name, n_params in component_n_params.items():
            self.logger.info("{:16}: {:,}".format(name, n_params))
        self.logger.info(
            "{:16}: {:,}".format("Total", sum(component_n_params.values()))
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        indices=None,
        lengths=None,
        **kwargs,
    ):
        """
        Modified from huggingface implementation to handle indices and lengths.
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "indices": indices,  # our custom kwargs
            "lengths": lengths,  # our custom kwargs
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        We need to resize as B x (N * L) instead of (B * N) x L here
        because the T5 forward method uses the input tensors to infer dimensions used in the decoder.
        EncoderWrapper resizes the inputs as (B * N) x L.
        """

        indices = kwargs["indices"]
        kwargs.pop("indices")
        lengths = kwargs["lengths"]
        kwargs.pop("lengths")
        kwargs["return_dict"] = False
        human_forecasts = kwargs.pop("human_forecasts", None)
        forecast_time_orders = kwargs.pop("forecast_time_orders", None)
        loss_reweight = kwargs.pop("loss_reweight", None)
        if loss_reweight is not None:
            loss_reweight_tf, loss_reweight_mc, loss_reweight_num = (
                loss_reweight["t/f"],
                loss_reweight["mc"],
                loss_reweight["num"],
            )
        else:
            loss_reweight_tf, loss_reweight_mc, loss_reweight_num = 1.0, 1.0, 1.0

        # inputs might have already be resized in the generate method
        if input_ids is not None:
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(
                input_ids.size(0), -1
            )  # [B, N=10, L=512] -> [B, N * L]

        if attention_mask != None:
            attention_mask = attention_mask.view(
                attention_mask.size(0), -1
            )  # [B, N=10, L=512] -> [B, N * L]

        indices_tf = indices[0][: lengths[0]]
        indices_mc = indices[1][: lengths[1]]
        indices_num = indices[2][: lengths[2]]
        labels_tf, labels_mc, labels_num = None, None, None
        batch_size = lengths.sum().item()
        n_passages = self.encoder.n_passages

        if labels is None:
            # inference mode
            kwargs["output_hidden_states"] = True
            decoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
            encoder_outputs = None
            if isinstance(decoder_outputs[2], tuple):
                hidden_state = decoder_outputs[2][
                    -1
                ]  # decoder output at the last layer
            elif isinstance(decoder_outputs[2], torch.Tensor):
                hidden_state = decoder_outputs[2]
            else:
                raise ValueError(
                    "Unknown type of decoder_outputs[2]: {}".format(
                        type(decoder_outputs[2])
                    )
                )
            previous_outputs = decoder_outputs[1]
            logits = decoder_outputs[0]
        else:
            # training mode
            labels_tf = torch.index_select(labels, 0, indices_tf).to(torch.int64)
            labels_mc = torch.index_select(labels, 0, indices_mc).to(torch.int64)
            labels_num = torch.index_select(labels, 0, indices_num).to(torch.int64)

            decoder_labels = copy.deepcopy(labels).to(torch.int64)
            decoder_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=decoder_labels,
                output_hidden_states=True,
                **kwargs,
            )
            encoder_outputs = decoder_outputs[-1][
                -1
            ]  # encoder output at the last layer  # [B * N, L, D]
            encoder_outputs = encoder_outputs.view(
                batch_size, n_passages, -1, encoder_outputs.size(-1)
            )  # [B, N, L, D]
            encoder_outputs = encoder_outputs * attention_mask.reshape(
                batch_size, n_passages, -1, 1
            )  # mask out padding
            encoder_outputs = encoder_outputs.mean(
                dim=2
            )  # average over the #tokens-per-passage dimension, [B, N, D]

            hidden_state = decoder_outputs[3][-1]  # decoder output at the last layer
            previous_outputs = decoder_outputs[2]
            logits = decoder_outputs[1]

        logits_tf = torch.index_select(logits, 0, indices_tf).view(-1, logits.size(-1))
        logits_mc = torch.index_select(logits, 0, indices_mc).view(-1, logits.size(-1))
        logits_num = torch.index_select(logits, 0, indices_num).view(
            -1, logits.size(-1)
        )

        if labels is None:
            if not self.training:
                # build Seq2SeqLMOutput type of outputs to adapt to later huggingface transformers framework
                # likely in generation mode
                return Seq2SeqLMOutput(loss=None, logits=logits, past_key_values=None)
            else:
                # old tuple type of outputs
                return (
                    logits,
                    previous_outputs,
                    logits_tf,
                    logits_mc,
                    logits_num,
                    hidden_state,
                )

        assert labels is not None
        loss_reweight_coef = torch.ones(
            batch_size, dtype=torch.float, device=labels.device
        )
        loss_reweight_coef[indices_tf] = loss_reweight_tf
        loss_reweight_coef[indices_mc] = loss_reweight_mc
        loss_reweight_coef[indices_num] = loss_reweight_num
        loss_reweight_coef = loss_reweight_coef[:, None].expand(-1, labels.size(-1))

        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1).to(torch.int64)

        # task loss
        loss_training = F.cross_entropy(
            logits, labels.view(-1), ignore_index=-100, reduction="none"
        )  # [B * 10]
        loss_training = (loss_training * loss_reweight_coef.reshape(-1)).mean()
        loss_tf = F.cross_entropy(
            logits_tf, labels_tf.view(-1), ignore_index=-100, reduction="mean"
        )
        loss_mc = F.cross_entropy(
            logits_mc, labels_mc.view(-1), ignore_index=-100, reduction="mean"
        )
        loss_num = F.cross_entropy(
            logits_num, labels_num.view(-1), ignore_index=-100, reduction="mean"
        )

        # human annotation regualrization loss
        # create causal mask based on context_time
        nheads = self.aln_tf.layers[0].self_attn.num_heads  # fixed
        visible_mask = (
            forecast_time_orders[:, :, None] >= forecast_time_orders[:, None, :]
        )  # [B, N, N]
        causal_mask = torch.zeros(
            batch_size,
            n_passages,
            n_passages,
            dtype=torch.float,
            device=visible_mask.device,
        )  # [B, N, N]
        causal_mask.masked_fill_(visible_mask == False, float("-inf"))  # [B, N, N]
        causal_mask = causal_mask.view(batch_size, 1, n_passages, n_passages).repeat(
            1, nheads, 1, 1
        )  # [B, H, N, N]
        causal_mask = causal_mask.view(
            batch_size * nheads, n_passages, n_passages
        )  # [B * H, N, N]

        aln_state = self.aln_tf(encoder_outputs, mask=causal_mask)  # [B, N, D]
        aln_out = self.aln_mlp(aln_state).squeeze(-1)  # [B, N]

        # compute alignment loss based on cross-entropy (equivalent to BCE here)
        h_forecast_weight = (human_forecasts != -100).float()
        loss_aln_tf = F.binary_cross_entropy_with_logits(
            aln_out[indices_tf],
            human_forecasts[indices_tf],
            reduction="mean",
            pos_weight=h_forecast_weight[indices_tf],
        )
        loss_aln_mc = F.binary_cross_entropy_with_logits(
            aln_out[indices_mc],
            human_forecasts[indices_mc],
            reduction="mean",
            pos_weight=h_forecast_weight[indices_mc],
        )
        loss_aln_num = (
            torch.abs(aln_out[indices_num] - human_forecasts[indices_num])
            * h_forecast_weight[indices_num]
        ).mean()

        # remove nan loss for cleaner logging
        loss_tf = torch.nan_to_num(loss_tf, nan=0.0)
        loss_mc = torch.nan_to_num(loss_mc, nan=0.0)
        loss_num = torch.nan_to_num(loss_num, nan=0.0)
        loss_aln_tf = torch.nan_to_num(loss_aln_tf, nan=0.0)
        loss_aln_mc = torch.nan_to_num(loss_aln_mc, nan=0.0)
        loss_aln_num = torch.nan_to_num(loss_aln_num, nan=0.0)

        loss_aln = loss_aln_tf + loss_aln_mc + loss_aln_num
        loss_training = loss_training + loss_aln * 0.1
        loss_training = torch.nan_to_num(loss_training, nan=0.0)

        acc_tf, acc_mc, acc_num = (
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        n_answer = int(labels.numel() / batch_size)
        if len(indices_tf):
            acc_tf = (
                logits_tf.reshape(len(indices_tf), n_answer, -1)[:, 0:2, :].argmax(-1)
                == labels_tf[:, 0:2]
            ).sum(dim=-1) == 2
            acc_tf = acc_tf.float().mean()
        if len(indices_mc):
            acc_mc = (
                logits_mc.reshape(len(indices_mc), n_answer, -1)[:, 0:2, :].argmax(-1)
                == labels_mc[:, 0:2]
            ).sum(dim=-1) == 2
            acc_mc = acc_mc.float().mean()
        if len(indices_num):
            acc_num = (
                logits_num.reshape(len(indices_num), n_answer, -1)[:, 0:2, :].argmax(-1)
                == labels_num[:, 0:2]
            ).sum(dim=-1) == 2
            acc_num = acc_num.float().mean()
        acc_tf = torch.nan_to_num(acc_tf, nan=0.0)
        acc_mc = torch.nan_to_num(acc_mc, nan=0.0)
        acc_num = torch.nan_to_num(acc_num, nan=0.0)

        loss_per_cat = (loss_tf, loss_mc, loss_num)
        loss_aln_per_cat = (loss_aln_tf, loss_aln_mc, loss_aln_num)
        acc_per_cat = (acc_tf, acc_mc, acc_num)

        return (
            logits,
            previous_outputs,
            loss_training,
            loss_per_cat,
            loss_aln_per_cat,
            acc_per_cat,
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        indices = kwargs["indices"]
        lengths = kwargs["lengths"]

        # note our self.encoder is a wrapper, the generation inference behavior must be adapted accordingly
        output = super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs,
        )
        return output

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5_multihead(self, state_dict):
        self.unwrap_encoder()
        own_state = self.state_dict()
        for name, _ in own_state.items():
            if name not in state_dict:
                continue
            if isinstance(state_dict[name], torch.nn.parameter.Parameter):
                state_dict[name] = state_dict[name].data
            own_state[name].copy_(state_dict[name])
        own_state_keys, load_state_keys = set(own_state.keys()), set(state_dict.keys())
        unique_own_state_keys = own_state_keys - load_state_keys
        unique_load_state_keys = load_state_keys - own_state_keys
        self.logger.info(
            "Our model has the following unique modules. They are not loaded from the pre-trained weights."
        )
        if len(unique_own_state_keys):
            [self.logger.info(key) for key in unique_own_state_keys]
        else:
            self.logger.info("None")
        self.logger.info(
            "The pre-trained weights have the following unique modules. They are not loaded into our model."
        )
        if len(unique_load_state_keys):
            [self.logger.info(key) for key in unique_load_state_keys]
        else:
            self.logger.info("None")
        self.wrap_encoder()

        for parameter in self.parameters():
            parameter.requires_grad = True

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def check_param(self, module_params):
        model_params = list(self.parameters())
        for param in module_params:
            if any(
                [
                    (param == model_p).all()
                    for model_p in model_params
                    if param.shape == model_p.shape
                ]
            ):
                self.logger.info("True")


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        self.apply_checkpoint_wrapper(self.encoder, use_checkpoint)

        self.main_input_name = (
            self.encoder.main_input_name
        )  # fix generate method bug in latest huggingface transformers framework

    def apply_checkpoint_wrapper(self, t5stack, use_checkpoint):
        """
        Wrap each block of the encoder to enable checkpointing.
        """
        block = []
        for mod in t5stack.block:
            wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
            block.append(wrapped_mod)
        block = nn.ModuleList(block)
        t5stack.block = block

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        # handle special kwargs: indices, lengths
        encoder_kwargs = copy.deepcopy(kwargs)
        if "indices" in encoder_kwargs:
            encoder_kwargs.pop("indices")
        if "lengths" in encoder_kwargs:
            encoder_kwargs.pop("lengths")
        kwargs = encoder_kwargs

        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (
            outputs[0].view(bsz, self.n_passages * passage_length, -1),
        ) + outputs[1:]
        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [], dtype=torch.float, device=output[0].device, requires_grad=True
                )
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

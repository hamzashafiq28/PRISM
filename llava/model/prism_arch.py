from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from llava.constants import IGNORE_INDEX, DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, N_ECG_TOKENS, N_EHR_TOKENS, N_HIST_TOKENS
from .multimodal_encoder.builder import build_ecg_tower, build_ehr_tower, build_history_tower
from .multimodal_projector.builder import build_soft_token_projector

class PrismMetaModel:

    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, 'ecg_tower'):
            self.ecg_tower = None
            self.ehr_tower = None
            self.hist_tower = None
            self.ecg_projector = None
            self.ehr_projector = None
            self.hist_projector = None

    def get_ecg_tower(self):
        return getattr(self, 'ecg_tower', None)

    def get_ehr_tower(self):
        return getattr(self, 'ehr_tower', None)

    def get_history_tower(self):
        return getattr(self, 'hist_tower', None)

    def initialize_medical_modules(self, model_args) -> None:
        if self.get_ecg_tower() is None:
            self.ecg_tower = build_ecg_tower(model_args, delay_load=False)
        else:
            self.ecg_tower.load_model()
        if self.get_ehr_tower() is None:
            self.ehr_tower = build_ehr_tower(model_args)
        if self.get_history_tower() is None:
            self.hist_tower = build_history_tower(model_args)
        d_llm = getattr(model_args, 'd_llm', 2560)
        hidden = getattr(model_args, 'proj_hidden', 1024)
        repr_dim = getattr(model_args, 'repr_dim', 512)
        ecg_dim = getattr(model_args, 'ecg_dim', self.ecg_tower.hidden_size)
        n_ecg = getattr(model_args, 'n_ecg_tokens', N_ECG_TOKENS)
        n_ehr = getattr(model_args, 'n_ehr_tokens', N_EHR_TOKENS)
        n_hist = getattr(model_args, 'n_hist_tokens', N_HIST_TOKENS)
        if getattr(self, 'ecg_projector', None) is None:
            self.ecg_projector = build_soft_token_projector(ecg_dim, d_llm, n_ecg, hidden)
        if getattr(self, 'ehr_projector', None) is None:
            self.ehr_projector = build_soft_token_projector(repr_dim, d_llm, n_ehr, hidden)
        if getattr(self, 'hist_projector', None) is None:
            self.hist_projector = build_soft_token_projector(repr_dim, d_llm, n_hist, hidden)
        self.config.ecg_tower = getattr(model_args, 'ecg_tower', '')
        self.config.ecg_cfg = getattr(model_args, 'ecg_cfg', '')
        self.config.ecg_dim = ecg_dim
        self.config.repr_dim = repr_dim
        self.config.d_llm = d_llm
        self.config.n_ecg_tokens = n_ecg
        self.config.n_ehr_tokens = n_ehr
        self.config.n_hist_tokens = n_hist
        pass

    def load_medical_projectors(self, ckpt_path: str) -> None:
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        proj_state = {k: v for k, v in state.items() if k.startswith(('ecg_projector', 'ehr_projector', 'hist_projector'))}
        missing, unexpected = self.load_state_dict(proj_state, strict=False)
        pass

class PrismMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_ecg_tower(self):
        return self.get_model().get_ecg_tower()

    def get_ehr_tower(self):
        return self.get_model().get_ehr_tower()

    def get_history_tower(self):
        return self.get_model().get_history_tower()

    def encode_ecg(self, waveforms: torch.Tensor) -> torch.Tensor:
        feats = self.get_ecg_tower()(waveforms)
        return self.get_model().ecg_projector(feats)

    def encode_ehr(self, ehr_feat: torch.Tensor) -> torch.Tensor:
        feats = self.get_ehr_tower()(ehr_feat)
        return self.get_model().ehr_projector(feats)

    def encode_ecg_history(self, ecg_emb_list: list, days_list: list) -> torch.Tensor:
        feats = self.get_history_tower().forward_batch(ecg_emb_list, days_list)
        return self.get_model().hist_projector(feats)

    def _get_embed_fn(self):
        m = self.get_model()
        if hasattr(m, 'embed_tokens'):
            return m.embed_tokens
        if hasattr(m, 'model') and hasattr(m.model, 'embed_tokens'):
            return m.model.embed_tokens
        raise AttributeError('Cannot locate embed_tokens in model hierarchy.')

    def prepare_inputs_labels_for_multimodal(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, ecg_soft: torch.Tensor, ehr_soft: torch.Tensor, hist_soft: torch.Tensor, ecg_token_id: int, ehr_token_id: int, hist_token_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed_fn = self._get_embed_fn()
        B = input_ids.size(0)
        special_ids = {ecg_token_id, ehr_token_id, hist_token_id}
        tok_map = {ecg_token_id: ecg_soft, ehr_token_id: ehr_soft, hist_token_id: hist_soft}
        new_embeds_list = []
        new_labels_list = []
        new_attn_list = []
        for b in range(B):
            ids = input_ids[b]
            lbs = labels[b]
            attn = attention_mask[b]
            cur_embed = []
            cur_labels = []
            cur_attn = []
            i = 0
            while i < len(ids):
                tok_id = ids[i].item()
                if tok_id in special_ids:
                    soft = tok_map[tok_id]
                    n = soft.size(1)
                    cur_embed.append(soft[b])
                    cur_labels.extend([IGNORE_INDEX] * n)
                    cur_attn.extend([1] * n)
                    i += 1
                else:
                    j = i
                    while j < len(ids) and ids[j].item() not in special_ids:
                        j += 1
                    chunk = ids[i:j]
                    cur_embed.append(embed_fn(chunk))
                    cur_labels.extend(lbs[i:j].tolist())
                    cur_attn.extend(attn[i:j].tolist())
                    i = j
            new_embeds_list.append(torch.cat(cur_embed, dim=0))
            new_labels_list.append(torch.tensor(cur_labels, dtype=torch.long, device=input_ids.device))
            new_attn_list.append(torch.tensor(cur_attn, dtype=attention_mask.dtype, device=input_ids.device))
        max_len = max((e.size(0) for e in new_embeds_list))
        d_llm = new_embeds_list[0].size(1)
        padded_embeds = torch.zeros(B, max_len, d_llm, dtype=new_embeds_list[0].dtype, device=input_ids.device)
        padded_labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=input_ids.device)
        padded_attn = torch.zeros(B, max_len, dtype=attention_mask.dtype, device=input_ids.device)
        for b in range(B):
            L = new_embeds_list[b].size(0)
            padded_embeds[b, :L] = new_embeds_list[b]
            padded_labels[b, :L] = new_labels_list[b]
            padded_attn[b, :L] = new_attn_list[b]
        return (padded_embeds, padded_labels, padded_attn)

    def initialize_medical_tokenizer(self, model_args, tokenizer) -> None:
        new_tokens = [DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN]
        n_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
        if n_added > 0:
            self.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                inp = self.get_input_embeddings().weight.data
                out = self.get_output_embeddings().weight.data
                avg_in = inp[:-n_added].mean(0, keepdim=True)
                avg_out = out[:-n_added].mean(0, keepdim=True)
                inp[-n_added:] = avg_in.expand(n_added, -1)
                out[-n_added:] = avg_out.expand(n_added, -1)
        ecg_tid = tokenizer.convert_tokens_to_ids(DEFAULT_ECG_TOKEN)
        ehr_tid = tokenizer.convert_tokens_to_ids(DEFAULT_EHR_TOKEN)
        hist_tid = tokenizer.convert_tokens_to_ids(DEFAULT_ECG_HIST_TOKEN)
        self.config.ecg_token_id = ecg_tid
        self.config.ehr_token_id = ehr_tid
        self.config.hist_token_id = hist_tid
        pass

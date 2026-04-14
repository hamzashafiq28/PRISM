from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from llava.constants import IGNORE_INDEX, DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, N_ECG_TOKENS, N_EHR_TOKENS, N_HIST_TOKENS
from llava.model.multimodal_encoder.builder import build_ecg_tower, build_ehr_tower, build_history_tower
from llava.model.multimodal_projector.builder import build_soft_token_projector
from llava.model.prism_arch import PrismMetaForCausalLM
try:
    from transformers import Gemma3ForConditionalGeneration
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    pass

@dataclass
class PrismModelArgs:
    medgemma: str = 'google/medgemma-4b-it'
    ecg_tower: str = '/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder/models/best.pt'
    ecg_cfg: str = '/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder/configs/config0.json'
    ecg_dim: int = 256
    feature_cols: List[str] = field(default_factory=list)
    cat_feature_cols: List[str] = field(default_factory=list)
    cat_cardinalities: List[int] = field(default_factory=list)
    repr_dim: int = 512
    ehr_d_model: int = 192
    ehr_n_heads: int = 4
    ehr_n_layers: int = 3
    ehr_dropout: float = 0.1
    n_ecg_tokens: int = N_ECG_TOKENS
    n_ehr_tokens: int = N_EHR_TOKENS
    n_hist_tokens: int = N_HIST_TOKENS
    proj_hidden: int = 1024
    d_llm: int = 2560

class PrismMedGemma(nn.Module, PrismMetaForCausalLM):

    def __init__(self, model_args: PrismModelArgs, tokenizer, bf16: bool=True):
        nn.Module.__init__(self)
        self.model_args = model_args
        self.tokenizer = tokenizer
        assert _HF_AVAILABLE, 'transformers with Gemma3 support is required.'
        pass
        _load_kwargs = dict(torch_dtype=torch.bfloat16 if bf16 else torch.float32, device_map='cpu')
        try:
            import flash_attn as _fa_mod
            _fa_ver = getattr(_fa_mod, '__version__', 'unknown')
            _fa_available = True
        except ImportError:
            _fa_ver = None
            _fa_available = False
        if _fa_available:
            try:
                _full = Gemma3ForConditionalGeneration.from_pretrained(model_args.medgemma, attn_implementation='flash_attention_2', **_load_kwargs)
                pass
            except Exception as _fa_err:
                pass
                pass
                _full = Gemma3ForConditionalGeneration.from_pretrained(model_args.medgemma, **_load_kwargs)
        else:
            pass
            pass
            _full = Gemma3ForConditionalGeneration.from_pretrained(model_args.medgemma, **_load_kwargs)
        self.language_model = _full.language_model
        del _full
        torch.cuda.empty_cache()
        d_llm = self.language_model.config.hidden_size
        model_args.d_llm = d_llm
        pass
        self.language_model.requires_grad_(False)
        self.ecg_tower = build_ecg_tower(model_args, delay_load=False)
        model_args.ecg_dim = self.ecg_tower.hidden_size
        pass
        self.ehr_tower = build_ehr_tower(model_args)
        self.hist_tower = build_history_tower(model_args)
        self.ecg_projector = build_soft_token_projector(model_args.ecg_dim, d_llm, model_args.n_ecg_tokens, model_args.proj_hidden)
        self.ehr_projector = build_soft_token_projector(model_args.repr_dim, d_llm, model_args.n_ehr_tokens, model_args.proj_hidden)
        self.hist_projector = build_soft_token_projector(model_args.repr_dim, d_llm, model_args.n_hist_tokens, model_args.proj_hidden)
        self.language_model.resize_token_embeddings(len(tokenizer))
        self._init_new_token_embeddings(len(tokenizer))
        self.ecg_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_ECG_TOKEN)
        self.ehr_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EHR_TOKEN)
        self.hist_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_ECG_HIST_TOKEN)
        pass

    def get_model(self):
        return self

    def get_ecg_tower(self):
        return self.ecg_tower

    def get_ehr_tower(self):
        return self.ehr_tower

    def get_history_tower(self):
        return self.hist_tower

    def get_embed_tokens(self) -> nn.Embedding:
        lm = self.language_model
        if hasattr(lm, 'base_model') and hasattr(lm.base_model, 'model'):
            lm = lm.base_model.model
        if hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
            return lm.model.embed_tokens
        if hasattr(lm, 'embed_tokens'):
            return lm.embed_tokens
        raise AttributeError('Cannot find embed_tokens in language_model.')

    def _get_embed_fn(self):
        return self.get_embed_tokens()

    def _init_new_token_embeddings(self, new_vocab_size: int) -> None:
        n_added = new_vocab_size - self.language_model.config.vocab_size
        if n_added <= 0:
            return
        with torch.no_grad():
            inp = self.language_model.get_input_embeddings().weight.data
            out = self.language_model.get_output_embeddings().weight.data
            inp[-n_added:] = inp[:-n_added].mean(0, keepdim=True).expand(n_added, -1)
            out[-n_added:] = out[:-n_added].mean(0, keepdim=True).expand(n_added, -1)

    def resize_token_embeddings(self, new_size: int):
        self.language_model.resize_token_embeddings(new_size)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def encode_ecg(self, waveforms: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.ecg_tower(waveforms)
        return self.ecg_projector(feats.to(self.ecg_projector.net[0].weight.dtype))

    def encode_ehr(self, ehr_feat: torch.Tensor) -> torch.Tensor:
        feats = self.ehr_tower(ehr_feat)
        return self.ehr_projector(feats)

    def encode_ecg_history(self, ecg_emb_list: list, days_list: list) -> torch.Tensor:
        feats = self.hist_tower.forward_batch(ecg_emb_list, days_list)
        return self.hist_projector(feats)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, ecg_waveforms: torch.Tensor, ehr_features: torch.Tensor, ecg_emb_list: list, days_list: list):
        ecg_soft = self.encode_ecg(ecg_waveforms)
        ehr_soft = self.encode_ehr(ehr_features)
        hist_soft = self.encode_ecg_history(ecg_emb_list, days_list)
        new_embeds, new_labels, new_attn = self.prepare_inputs_labels_for_multimodal(input_ids=input_ids, attention_mask=attention_mask, labels=labels, ecg_soft=ecg_soft, ehr_soft=ehr_soft, hist_soft=hist_soft, ecg_token_id=self.ecg_token_id, ehr_token_id=self.ehr_token_id, hist_token_id=self.hist_token_id)
        output = self.language_model(inputs_embeds=new_embeds, attention_mask=new_attn, labels=new_labels)
        return output

    def print_trainable_params(self) -> None:
        total = trainable = 0
        for name, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        pass

    def save_stage1_checkpoint(self, path: str, extra: dict=None) -> None:
        state: Dict = {}
        for name, p in self.named_parameters():
            if p.requires_grad:
                state[name] = p.detach().cpu()
        if extra:
            state.update(extra)
        torch.save(state, path)
        pass

    @classmethod
    def load_stage1_checkpoint(cls, model: 'PrismMedGemma', path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        own_keys = {n for n, p in model.named_parameters() if p.requires_grad}
        filtered = {k: v for k, v in ckpt.items() if k in own_keys}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        pass

from __future__ import annotations
import os
import sys
import importlib.util
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

class ECGFounderTower(nn.Module):
    _HIDDEN_SIZE = 1024

    def __init__(self, ckpt_path: str, net1d_path: Optional[str]=None):
        super().__init__()
        self._hidden_size = self._HIDDEN_SIZE
        Net1D = self._import_net1d(net1d_path)
        self.encoder = Net1D(in_channels=12, base_filters=64, ratio=1, filter_list=[64, 160, 160, 400, 400, 1024, 1024], m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2, groups_width=16, verbose=False, use_bn=False, use_do=False, n_classes=150, return_features=True)
        pass
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')}
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        pass
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @staticmethod
    def _import_net1d(net1d_path: Optional[str]=None):
        if net1d_path and os.path.isfile(net1d_path):
            spec = importlib.util.spec_from_file_location('net1d', net1d_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.Net1D
        try:
            from net1d import Net1D
            return Net1D
        except ImportError:
            pass
        raise ImportError('Cannot import Net1D. Provide --net1d_path pointing to ECGFounder/net1d.py or install it on sys.path.')

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, deep_features = self.encoder(x)
        return deep_features

class ESIConvNeXtTower(nn.Module):
    VARIANT_DIMS = {'base': 1024, 'tiny': 768, 'nano': 640, 'atto': 320, 'large': 1536}

    def __init__(self, ckpt_path: str, variant: str='base', convnextv2_path: Optional[str]=None):
        super().__init__()
        assert variant in self.VARIANT_DIMS, f"Unknown variant '{variant}'. Choose from {list(self.VARIANT_DIMS)}"
        self._hidden_size = self.VARIANT_DIMS[variant]
        builder_fn = self._import_builder(variant, convnextv2_path)
        self.encoder = builder_fn(in_chans=12, num_classes=1000, return_embedding=True)
        pass
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if any((k.startswith('module.') for k in state_dict)):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        pass
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @staticmethod
    def _import_builder(variant: str, convnextv2_path: Optional[str]=None):
        fn_name = f'convnextv2_{variant}'
        if convnextv2_path and os.path.isfile(convnextv2_path):
            spec = importlib.util.spec_from_file_location('convnextv2', convnextv2_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, fn_name)
        try:
            from model.convnextv2 import ConvNeXtV2
            import model.convnextv2 as cv2_mod
            return getattr(cv2_mod, fn_name)
        except ImportError:
            pass
        raise ImportError(f'Cannot import {fn_name}. Provide --convnextv2_path pointing to ESI/model/convnextv2.py or add ESI to sys.path.')

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.encoder(x)

class HeartBERTTower(nn.Module):
    _HIDDEN_SIZE = 768

    def __init__(self, model_name: str='tianyudizhua1/HeartBert', max_length: int=512, bert_encode_chunk_size: int=0):
        super().__init__()
        self._hidden_size = self._HIDDEN_SIZE
        self.max_length = max_length
        self.bert_encode_chunk_size = bert_encode_chunk_size
        from transformers import AutoTokenizer, AutoModel
        pass
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.bert_model.requires_grad_(False)
        self.bert_model.eval()

    @staticmethod
    def _quantize_signal(signal_1d: np.ndarray, alphabet_size: int=100) -> str:
        diffs = np.diff(signal_1d)
        diffs = np.insert(diffs, 0, 0.0)
        sorted_arr = np.sort(diffs)
        n = len(sorted_arr)
        breakpoints = []
        for i in range(alphabet_size):
            idx = min(int((i + 1) / alphabet_size * n) - 1, n - 1)
            breakpoints.append(sorted_arr[idx])
        breakpoints[-1] = 1e+100
        chars = []
        for v in diffs:
            for j, bp in enumerate(breakpoints):
                if v < bp:
                    chars.append(chr(65 + j))
                    break
        return ''.join(chars)

    @property
    def hidden_size(self) -> int:
        return self._HIDDEN_SIZE

    @property
    def dtype(self):
        return next(self.bert_model.parameters()).dtype

    @property
    def device(self):
        return next(self.bert_model.parameters()).device

    def _lead_to_text(self, sig_1d: np.ndarray) -> str:
        sig_min, sig_max = (sig_1d.min(), sig_1d.max())
        if sig_max - sig_min > 1e-08:
            sig = (sig_1d - sig_min) / (sig_max - sig_min)
        else:
            sig = np.zeros_like(sig_1d)
        return self._quantize_signal(sig, alphabet_size=100)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        device = x.device
        x_cpu = x.detach().cpu().numpy()
        texts: list[str] = []
        for b in range(B):
            for c in range(C):
                sig = x_cpu[b, c].astype(np.float64, copy=False)
                texts.append(self._lead_to_text(sig))
        n_seq = len(texts)
        bes = self.bert_encode_chunk_size
        chunk = min(bes, n_seq) if bes and bes > 0 else n_seq
        cls_chunks: list[torch.Tensor] = []
        for start in range(0, n_seq, chunk):
            batch_texts = texts[start:start + chunk]
            tokens = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            attn_mask = tokens['attention_mask'].to(device)
            out = self.bert_model(input_ids=input_ids, attention_mask=attn_mask)
            cls_chunks.append(out.last_hidden_state[:, 0, :])
        per_lead_cls = torch.cat(cls_chunks, dim=0)
        return per_lead_cls.view(B, C, -1).mean(dim=1)
ENCODER_REGISTRY = {'ecgfounder': ECGFounderTower, 'esi_convnext_base': ESIConvNeXtTower, 'esi_convnext_tiny': ESIConvNeXtTower, 'esi_convnext_nano': ESIConvNeXtTower, 'heartbert': HeartBERTTower}
ENCODER_DIMS = {'ecgfounder': 1024, 'esi_convnext_base': 1024, 'esi_convnext_tiny': 768, 'esi_convnext_nano': 640, 'heartbert': 768}

def build_external_ecg_tower(name: str, **kwargs) -> nn.Module:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Choose from: {list(ENCODER_REGISTRY)}")
    cls = ENCODER_REGISTRY[name]
    if name == 'ecgfounder':
        return cls(ckpt_path=kwargs['ckpt_path'], net1d_path=kwargs.get('net1d_path'))
    elif name.startswith('esi_convnext'):
        variant = name.split('_')[-1]
        return cls(ckpt_path=kwargs['ckpt_path'], variant=variant, convnextv2_path=kwargs.get('convnextv2_path'))
    elif name == 'heartbert':
        return cls(model_name=kwargs.get('model_name', 'tianyudizhua1/HeartBert'), max_length=kwargs.get('max_length', 512), bert_encode_chunk_size=kwargs.get('bert_encode_chunk_size', 0))
    else:
        raise ValueError(f'Unhandled encoder: {name}')

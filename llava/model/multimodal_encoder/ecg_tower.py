from __future__ import annotations
import os
import sys
from typing import Optional
import torch
import torch.nn as nn
_CMELT_SEARCH_PATHS = ['/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder']

class ECGTower(nn.Module):

    def __init__(self, ecg_ckpt: str, ecg_cfg: str, delay_load: bool=False):
        super().__init__()
        self.is_loaded = False
        self.ecg_ckpt = ecg_ckpt
        self.ecg_cfg = ecg_cfg
        self._encoder: Optional[nn.Module] = None
        self._ecg_dim: int = 0
        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None) -> None:
        if self.is_loaded:
            pass
            return
        search_dirs = list(_CMELT_SEARCH_PATHS)
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'llava', 'model', 'ecg_encoder')
        search_dirs.insert(0, os.path.normpath(local_dir))
        for d in search_dirs:
            if os.path.isdir(d) and d not in sys.path:
                sys.path.insert(0, d)
        from extract_ecg_feature import ECGEncoder as _CMELT
        _orig_init = _CMELT.__init__
        _ckpt, _cfg = (self.ecg_ckpt, self.ecg_cfg)

        def _patched_init(self_inner, *a, **kw):
            _orig_init(self_inner, *a, **kw)
            self_inner.checkpoint_path = _ckpt
            self_inner.config_path = _cfg
            self_inner._load_model()
        _CMELT.__init__ = _patched_init
        enc = _CMELT()
        _CMELT.__init__ = _orig_init
        enc.requires_grad_(False)
        enc.eval()
        self._encoder = enc
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 5000)
            out = enc(dummy)
            if isinstance(out, (list, tuple)):
                out = out[0]
            self._ecg_dim = out.shape[-1]
        self.is_loaded = True
        pass

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            self.load_model()
        out = self._encoder(waveforms.to(device=self.device, dtype=self.dtype))
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 3:
            out = out[:, 0]
        return out

    @property
    def hidden_size(self) -> int:
        return self._ecg_dim

    @property
    def dtype(self) -> torch.dtype:
        if self._encoder is None:
            return torch.float32
        p = next(iter(self._encoder.parameters()), None)
        return p.dtype if p is not None else torch.float32

    @property
    def device(self) -> torch.device:
        if self._encoder is None:
            return torch.device('cpu')
        p = next(iter(self._encoder.parameters()), None)
        return p.device if p is not None else torch.device('cpu')

    @property
    def dummy_feature(self) -> torch.Tensor:
        return torch.zeros(1, self._ecg_dim, device=self.device, dtype=self.dtype)

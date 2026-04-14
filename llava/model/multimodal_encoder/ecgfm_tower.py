from __future__ import annotations
import importlib.util
import os
import sys
from typing import Optional
import torch
import torch.nn as nn

class ECGFMTower(nn.Module):

    def __init__(self, ecgfm_ckpt: str, ecgfm_repo_dir: str, delay_load: bool=False):
        super().__init__()
        self.is_loaded = False
        self.ecgfm_ckpt = ecgfm_ckpt
        self.ecgfm_repo_dir = ecgfm_repo_dir
        self._encoder: Optional[nn.Module] = None
        self._ecg_dim: int = 0
        if not delay_load:
            self.load_model()

    def _ensure_fastai_stubs(self) -> None:
        try:
            import fastai
            return
        except ImportError:
            pass
        import types
        import inspect
        import re
        core = types.ModuleType('fastai.core')
        core.Optional = Optional
        core.Collection = list

        def _listify(o):
            if o is None:
                return []
            if isinstance(o, (list, tuple)):
                return list(o)
            return [o]
        core.listify = _listify
        core.inspect = inspect
        core.re = re
        layers = types.ModuleType('fastai.layers')

        def _bn_drop_lin(n_in, n_out, bn=True, p=0.0, actn=None):
            ls = []
            if bn:
                ls.append(nn.BatchNorm1d(n_in))
            if p > 0:
                ls.append(nn.Dropout(p))
            ls.append(nn.Linear(n_in, n_out))
            if actn is not None:
                ls.append(actn)
            return ls

        class _Flatten(nn.Module):

            def forward(self, x):
                return x.view(x.size(0), -1)
        layers.bn_drop_lin = _bn_drop_lin
        layers.Flatten = _Flatten
        fastai_pkg = types.ModuleType('fastai')
        sys.modules['fastai'] = fastai_pkg
        sys.modules['fastai.core'] = core
        sys.modules['fastai.layers'] = layers

    def load_model(self, device_map=None) -> None:
        if self.is_loaded:
            pass
            return
        self._ensure_fastai_stubs()
        repo = self.ecgfm_repo_dir
        if repo not in sys.path:
            sys.path.insert(0, repo)
        mod_path = os.path.join(repo, 'models', 'xresnet1d_101.py')
        spec = importlib.util.spec_from_file_location('ecgfm_xresnet1d_101', mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        encoder = mod.xresnet1d101(num_classes=5, input_channels=12, kernel_size=5, ps_head=0.5, lin_ftrs_head=[768], use_ecgNet_Diagnosis='other')
        if self.ecgfm_ckpt and os.path.isfile(self.ecgfm_ckpt):
            ckpt = torch.load(self.ecgfm_ckpt, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and 'ecg_model' in ckpt:
                state = ckpt['ecg_model']
            else:
                state = ckpt
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            pass
            del ckpt
        else:
            pass
        encoder.requires_grad_(False)
        encoder.eval()
        self._encoder = encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 12, 5000)
            out = encoder(dummy)
            if out.ndim == 3:
                self._ecg_dim = out.shape[1]
            elif out.ndim == 2:
                self._ecg_dim = out.shape[-1]
            else:
                raise ValueError(f'Unexpected ECGFM output shape: {out.shape}')
        self.is_loaded = True
        pass

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            self.load_model()
        out = self._encoder(waveforms.to(device=self.device, dtype=self.dtype))
        if out.ndim == 3:
            out = out.mean(dim=-1)
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

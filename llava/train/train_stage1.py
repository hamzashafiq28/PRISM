from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
try:
    import faiss
    FAISS_OK = True
except Exception as _faiss_err:
    FAISS_OK = False
    pass
_LLAVA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _LLAVA_ROOT not in sys.path:
    sys.path.insert(0, _LLAVA_ROOT)
from llava.constants import DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, IGNORE_INDEX
from llava.model.language_model.prism_medgemma import PrismMedGemma, PrismModelArgs
try:
    from transformers import AutoTokenizer
except ImportError:
    raise RuntimeError('transformers is required: pip install transformers>=4.47')

@dataclass
class TrainConfig:
    train_json: str = '/projects/prjs1786/CNDP/New_Code/train_raft1.json'
    val_json: str = '/projects/prjs1786/CNDP/New_Code/test_raft1.json'
    train_val_csv: str = ''
    test_csv: str = ''
    patient_json: str = ''
    h5_root: str = ''
    ecg_cache: str = ''
    medgemma: str = 'google/medgemma-4b-it'
    ecg_tower: str = '/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder/models/best.pt'
    ecg_cfg: str = '/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder/configs/config0.json'
    output_dir: str = './stage1_out'
    resume: str = ''
    repr_dim: int = 512
    ehr_d_model: int = 192
    ehr_n_heads: int = 4
    ehr_n_layers: int = 3
    ehr_dropout: float = 0.1
    n_ecg_tokens: int = 8
    n_ehr_tokens: int = 8
    n_hist_tokens: int = 8
    proj_hidden: int = 1024
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 0.0002
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_samples: int = 0
    max_seq_len: int = 2048
    num_workers: int = 2
    seed: int = 42
    bf16: bool = True
    cls_alpha: float = 0.1
    topk: int = 50
    export_after_train: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class LazyJSONIndex:

    def __init__(self, path: str):
        self.path = path
        self.offsets = self._load_or_build_index()

    def _load_or_build_index(self) -> np.ndarray:
        idx_path = self.path + '.offsets.npy'
        if os.path.exists(idx_path):
            pass
            return np.load(idx_path)
        pass
        offsets = []
        with open(self.path, 'rb') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.lstrip()
                if stripped.startswith(b'{'):
                    offsets.append(offset)
        arr = np.array(offsets, dtype=np.int64)
        np.save(idx_path, arr)
        pass
        return arr

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> dict:
        with open(self.path, 'rb') as f:
            f.seek(int(self.offsets[idx]))
            line = f.readline().decode('utf-8').rstrip(',\n')
        return json.loads(line)
_VITAL_KEYS = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']

def build_ehr_matrix(train_val_csv: str, test_csv: str, patient_json: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[int]]:
    pass
    df_tv = pd.read_csv(train_val_csv, low_memory=False)
    df_test = pd.read_csv(test_csv, low_memory=False)
    for col in ('Unnamed: 0', 'index'):
        for df_ in (df_tv, df_test):
            if col in df_.columns:
                df_.drop(columns=[col], inplace=True)
    prefixes = ('biometrics_', 'demographics_', 'labvalues_', 'vitals_')
    feature_cols = [c for c in df_tv.columns if c.startswith(prefixes)]
    label_cols = [c for c in df_tv.columns if c.startswith('diagnoses_') or c.startswith('deterioration_')]
    for c in feature_cols + label_cols:
        if c not in df_test.columns:
            df_test[c] = 0.0
    base = [c for c in feature_cols if not c.endswith('_missing')]
    miss_tv = df_tv[base].isnull().astype(np.float32).rename(columns=lambda c: c + '_missing')
    miss_test = df_test[base].isnull().astype(np.float32).rename(columns=lambda c: c + '_missing')
    df_tv = pd.concat([df_tv, miss_tv], axis=1).copy()
    df_test = pd.concat([df_test, miss_test], axis=1).copy()
    feature_cols = feature_cols + [c + '_missing' for c in base]
    mask = df_tv['general_strat_fold'] < 18 if 'general_strat_fold' in df_tv.columns else pd.Series(True, index=df_tv.index)
    medians = df_tv.loc[mask, base].median().to_dict()
    for c in base:
        df_tv[c] = df_tv[c].fillna(medians[c])
        df_test[c] = df_test[c].fillna(medians[c])
    cat_cols: List[str] = []
    cat_cards: List[int] = []
    for col in ('demographics_gender', 'vitals_acuity'):
        if col not in df_tv.columns:
            continue
        combined = pd.concat([df_tv[col], df_test[col]])
        codes, _ = pd.factorize(combined.fillna(-1).astype(int), sort=True)
        n = len(df_tv)
        df_tv[col] = codes[:n].astype(np.int64)
        df_test[col] = codes[n:].astype(np.int64)
        cat_cols.append(col)
        cat_cards.append(int(codes.max()) + 1)
    vitals_lookup = _extract_vitals_history(patient_json)
    if vitals_lookup:
        n_vhf = len(_VITAL_KEYS) * 7
        vhf_names = [f'vhf_{vk}_{s}' for vk in _VITAL_KEYS for s in ('mean', 'std', 'min', 'max', 'first', 'last', 'count')]

        def _vhf(df_):
            paths = df_['general_file_name'].astype(str).tolist() if 'general_file_name' in df_.columns else [''] * len(df_)
            mat = np.full((len(df_), n_vhf), np.nan, dtype=np.float32)
            for i, p in enumerate(paths):
                if p in vitals_lookup:
                    mat[i] = vitals_lookup[p]
            return mat
        vhf_tv = _vhf(df_tv)
        vhf_test = _vhf(df_test)
        vhf_med = np.nanmedian(vhf_tv[mask.values], axis=0)
        vhf_med = np.where(np.isnan(vhf_med), 0.0, vhf_med)
        for j in range(n_vhf):
            vhf_tv[np.isnan(vhf_tv[:, j]), j] = vhf_med[j]
            vhf_test[np.isnan(vhf_test[:, j]), j] = vhf_med[j]
        vhf_df_tv = pd.DataFrame(vhf_tv, columns=vhf_names, index=df_tv.index)
        vhf_df_test = pd.DataFrame(vhf_test, columns=vhf_names, index=df_test.index)
        df_tv = pd.concat([df_tv, vhf_df_tv], axis=1).copy()
        df_test = pd.concat([df_test, vhf_df_test], axis=1).copy()
        feature_cols = feature_cols + vhf_names
        pass
    trainval_mat = df_tv[feature_cols].values.astype(np.float32)
    test_mat = df_test[feature_cols].values.astype(np.float32)
    pass
    return (trainval_mat, test_mat, feature_cols, cat_cols, cat_cards)

def _extract_vitals_history(patient_json_path: str) -> Dict[str, np.ndarray]:
    if not patient_json_path or not os.path.exists(patient_json_path):
        return {}
    with open(patient_json_path) as f:
        pdata = json.load(f)
    lookup: Dict[str, np.ndarray] = {}
    n = len(_VITAL_KEYS) * 7
    for entry in pdata['patients'].values():
        ecg_path = entry.get('reference_ecg_path', '')
        vlist = entry.get('vitals_48h', [])
        feat = np.full(n, np.nan, dtype=np.float32)
        for vi, vkey in enumerate(_VITAL_KEYS):
            vals = [v[vkey] for v in vlist if v.get(vkey) is not None and (not math.isnan(v[vkey]))]
            if not vals:
                continue
            arr = np.array(vals, dtype=np.float32)
            base = vi * 7
            feat[base:base + 7] = [arr.mean(), arr.std() if len(arr) > 1 else 0.0, arr.min(), arr.max(), arr[0], arr[-1], min(len(arr), 100) / 100.0]
        lookup[ecg_path] = feat
    return lookup

def _extract_prior_ecg_index(patient_json_path: str) -> Dict[str, List[Dict]]:
    if not patient_json_path or not os.path.exists(patient_json_path):
        return {}
    with open(patient_json_path) as f:
        pdata = json.load(f)
    index: Dict[str, List[Dict]] = {}
    for entry in pdata['patients'].values():
        ref = entry.get('reference_ecg_path', '')
        prev = entry.get('previous_ecgs_48h', [])
        recs = [{'ecg_path': e['ecg_path'], 'days_before': abs(float(e['time_hours'])) / 24.0} for e in prev if e.get('ecg_path') and e.get('time_hours') is not None]
        if recs:
            index[ref] = recs
    return index
_ENGLISH_PREFIX = 'Respond in English only.\n'

def _extract_cls_label(answer: str) -> int:
    low = answer.lower()
    m = re.search('answer\\s*:\\s*(yes|no)', low)
    if m:
        return 1 if m.group(1) == 'yes' else 0
    for word in reversed(low.split()):
        w = word.strip('.,!?')
        if w == 'yes':
            return 1
        if w == 'no':
            return 0
    return -1
_REMOVE_BLOCKS = ('<start_of_image><retrived_ehr><end_of_image>', '<start_of_image><retrived_ecg_history><end_of_image>')

def preprocess_prompt(human_text: str) -> str:
    text = human_text
    for block in _REMOVE_BLOCKS:
        text = text.replace(block, '')
    return _ENGLISH_PREFIX + text

class ECGCache:

    def __init__(self, cache_dir: str):
        keys_path = os.path.join(cache_dir, 'ecg_keys.json')
        embs_path = os.path.join(cache_dir, 'ecg_embeddings.npy')
        if not os.path.exists(keys_path) or not os.path.exists(embs_path):
            raise FileNotFoundError(f'ECG cache not found in {cache_dir}. Run precompute_ecg_cache.py first.')
        with open(keys_path) as f:
            keys = json.load(f)
        self._embs = np.load(embs_path, mmap_mode='r')
        self._idx: Dict[str, int] = {k: i for i, k in enumerate(keys)}
        self.ecg_dim = self._embs.shape[1]
        pass

    def get(self, ecg_path: str) -> Optional[np.ndarray]:
        i = self._idx.get(ecg_path)
        if i is None:
            return None
        return self._embs[i].astype(np.float32)

    def __contains__(self, ecg_path: str) -> bool:
        return ecg_path in self._idx

def load_ecg_waveform(h5_root: str, ecg_path: str) -> Optional[np.ndarray]:
    if not ecg_path or not h5_root:
        return None
    full = os.path.join(h5_root, ecg_path) if not os.path.isabs(ecg_path) else ecg_path
    if not os.path.exists(full):
        full = full + '.h5'
        if not os.path.exists(full):
            return None
    try:
        with h5py.File(full, 'r') as f:
            key = list(f.keys())[0]
            waveform = f[key][()].astype(np.float32)
        if waveform.ndim == 2 and waveform.shape[1] == 12:
            waveform = waveform.T
        return waveform
    except Exception:
        return None

class Stage1Dataset(Dataset):

    def __init__(self, json_index: LazyJSONIndex, ehr_matrix: np.ndarray, prior_ecg_idx: Dict[str, List[Dict]]):
        self.index = json_index
        self.ehr_matrix = ehr_matrix
        self.prior_ecg_idx = prior_ecg_idx

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        ehr_row = entry['faiss_npy']['ehr_row']
        if ehr_row is None:
            ehr_feat = torch.zeros(self.ehr_matrix.shape[1], dtype=torch.float32)
        else:
            ehr_feat = torch.from_numpy(self.ehr_matrix[int(ehr_row)].copy())
        ecg_path = entry['ecg']
        has_history = entry['faiss_ecg_history'].get('has_history', False)
        prior_ecgs = self.prior_ecg_idx.get(ecg_path, []) if has_history else []
        human_text = entry['conversations'][0]['value']
        gpt_text = entry['conversations'][1]['value']
        prompt = preprocess_prompt(human_text)
        return {'ehr_feat': ehr_feat, 'ecg_path': ecg_path, 'prior_ecgs': prior_ecgs, 'has_history': has_history, 'prompt': prompt, 'answer': gpt_text, 'cls_label': _extract_cls_label(gpt_text)}

class CollateClass:
    _DUMMY_WAVE = torch.zeros(12, 5000)

    def __init__(self, h5_root: str, tokenizer, max_seq_len: int, ecg_cache: Optional[ECGCache]=None):
        self.h5_root = h5_root
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos = tokenizer.eos_token or ''
        self.ecg_cache = ecg_cache

    def _load_ecg_embedding(self, ecg_path: str) -> Optional[np.ndarray]:
        if self.ecg_cache is not None:
            return self.ecg_cache.get(ecg_path)
        return None

    def _load_ecg_waveform_or_dummy(self, ecg_path: str) -> torch.Tensor:
        w = load_ecg_waveform(self.h5_root, ecg_path)
        return torch.from_numpy(w) if w is not None else self._DUMMY_WAVE

    def __call__(self, batch: List[dict]) -> dict:
        B = len(batch)
        ehr_feat = torch.stack([item['ehr_feat'] for item in batch])
        cur_embs_list: List[Optional[np.ndarray]] = [self._load_ecg_embedding(item['ecg_path']) for item in batch]
        cache_hit = all((e is not None for e in cur_embs_list))
        if cache_hit:
            ecg_embeddings = torch.from_numpy(np.stack(cur_embs_list).astype(np.float32))
            ecg_waveforms = None
        else:
            waves = [self._load_ecg_waveform_or_dummy(item['ecg_path']) for item in batch]
            max_T = max((w.shape[-1] for w in waves))
            ecg_waveforms = torch.zeros(B, 12, max_T)
            for i, w in enumerate(waves):
                ecg_waveforms[i, :, :w.shape[-1]] = w
            ecg_embeddings = None
        prior_ecg_embs_list: List[Optional[torch.Tensor]] = []
        prior_ecg_waves_list: List[Optional[torch.Tensor]] = []
        days_list: List[Optional[torch.Tensor]] = []
        for item in batch:
            if not item['has_history'] or not item['prior_ecgs']:
                prior_ecg_embs_list.append(None)
                prior_ecg_waves_list.append(None)
                days_list.append(None)
                continue
            emb_list, wav_list, days = ([], [], [])
            for pe in item['prior_ecgs']:
                days.append(pe['days_before'])
                cached = self._load_ecg_embedding(pe['ecg_path'])
                if cached is not None:
                    emb_list.append(cached)
                    wav_list.append(None)
                else:
                    w = load_ecg_waveform(self.h5_root, pe['ecg_path'])
                    if w is None:
                        days.pop()
                        continue
                    emb_list.append(None)
                    wav_list.append(w)
            if not days:
                prior_ecg_embs_list.append(None)
                prior_ecg_waves_list.append(None)
                days_list.append(None)
                continue
            days_list.append(torch.tensor(days, dtype=torch.float32))
            if all((e is not None for e in emb_list)):
                prior_ecg_embs_list.append(torch.from_numpy(np.stack(emb_list).astype(np.float32)))
                prior_ecg_waves_list.append(None)
            else:
                prior_ecg_embs_list.append(None)
                pw_tensors = []
                for e, w in zip(emb_list, wav_list):
                    if e is not None:
                        pw_tensors.append(torch.from_numpy(np.zeros((12, 5000), dtype=np.float32)))
                    else:
                        pw_tensors.append(torch.from_numpy(w))
                max_T_p = max((pw.shape[-1] for pw in pw_tensors))
                padded = torch.zeros(len(pw_tensors), 12, max_T_p)
                for j, pw in enumerate(pw_tensors):
                    padded[j, :, :pw.shape[-1]] = pw
                prior_ecg_waves_list.append(padded)
        full_texts = [item['prompt'] + item['answer'] + self.eos for item in batch]
        prompt_texts = [item['prompt'] for item in batch]
        enc_full = self.tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_len)
        enc_prompt = self.tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_seq_len)
        input_ids = enc_full['input_ids']
        attention_mask = enc_full['attention_mask']
        labels = input_ids.clone()
        labels[attention_mask == 0] = IGNORE_INDEX
        for b in range(B):
            prompt_len = enc_prompt['attention_mask'][b].sum().item()
            labels[b, :prompt_len] = IGNORE_INDEX
        cls_labels = torch.tensor([item['cls_label'] for item in batch], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'ehr_feat': ehr_feat, 'cls_labels': cls_labels, 'ecg_embeddings': ecg_embeddings, 'ecg_waveforms': ecg_waveforms, 'prior_ecg_embs_list': prior_ecg_embs_list, 'prior_ecg_waves_list': prior_ecg_waves_list, 'days_list': days_list}

def get_lr(step: int, warmup: int, lr_max: float, total: int) -> float:
    if step < warmup:
        return lr_max * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * prog))

def _get_base_model(model) -> PrismMedGemma:
    return model.module if isinstance(model, DDP) else model

def run_epoch(model, loader: DataLoader, optimiser, scaler, cfg: TrainConfig, trainable, total_steps: int, global_step: int, train: bool=True, log_interval: int=100, ckpt_every: int=10000, ckpt_fn=None, cls_head: Optional[nn.Module]=None, cls_alpha: float=0.1, tokenizer=None) -> Tuple[float, int]:
    base = _get_base_model(model)
    if train:
        model.train()
        base.language_model.eval()
        base.ecg_tower.eval()
    else:
        model.eval()
    total_loss = 0.0
    n_batches = 0
    is_main = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    _shape_printed = [False]
    ctx = torch.cuda.amp.autocast(enabled=cfg.bf16 and cfg.device.startswith('cuda'), dtype=torch.bfloat16)
    with torch.set_grad_enabled(train):
        _iter = tqdm(loader, leave=False) if is_main else loader
        for step, batch in enumerate(_iter):
            ids = batch['input_ids'].to(cfg.device)
            attn = batch['attention_mask'].to(cfg.device)
            lbs = batch['labels'].to(cfg.device)
            ehr = batch['ehr_feat'].to(cfg.device)
            cls_labels = batch['cls_labels'].to(cfg.device)
            with torch.no_grad():
                if batch['ecg_embeddings'] is not None:
                    ecg_emb_cur = batch['ecg_embeddings'].to(cfg.device)
                    ecg_waves_or_emb = ecg_emb_cur
                    use_cached_ecg = True
                else:
                    ecg_waves_or_emb = batch['ecg_waveforms'].to(cfg.device)
                    use_cached_ecg = False
                ecg_emb_list: List[Optional[torch.Tensor]] = []
                days_l: List[Optional[torch.Tensor]] = []
                for pre_emb, pw, days in zip(batch['prior_ecg_embs_list'], batch['prior_ecg_waves_list'], batch['days_list']):
                    if pre_emb is not None:
                        ecg_emb_list.append(pre_emb.to(cfg.device))
                    elif pw is not None:
                        embs = base.ecg_tower(pw.to(cfg.device))
                        ecg_emb_list.append(embs)
                    else:
                        ecg_emb_list.append(None)
                    days_l.append(days.to(cfg.device) if days is not None else None)
            if is_main and (not _shape_printed[0]):
                _shape_printed[0] = True
                pass
                pass
                hist_present = [e for e in ecg_emb_list if e is not None]
                if hist_present:
                    pass
                else:
                    pass
                pass
                pass
            with ctx:
                if use_cached_ecg:
                    ecg_soft = base.ecg_projector(ecg_waves_or_emb.to(base.ecg_projector.net[0].weight.dtype))
                else:
                    with torch.no_grad():
                        ecg_feat = base.ecg_tower(ecg_waves_or_emb)
                    ecg_soft = base.ecg_projector(ecg_feat.to(base.ecg_projector.net[0].weight.dtype))
                ehr_soft = base.encode_ehr(ehr)
                hist_soft = base.encode_ecg_history(ecg_emb_list, days_l)
                new_embeds, new_labels, new_attn = base.prepare_inputs_labels_for_multimodal(ids, attn, lbs, ecg_soft, ehr_soft, hist_soft, base.ecg_token_id, base.ehr_token_id, base.hist_token_id)
                out = base.language_model(inputs_embeds=new_embeds, attention_mask=new_attn, labels=new_labels, output_hidden_states=cls_head is not None)
                loss = out.loss
                cls_loss_val = 0.0
                cls_acc_val = float('nan')
                if cls_head is not None:
                    hidden = out.hidden_states[-1]
                    B_ = hidden.size(0)
                    if tokenizer is not None:
                        _yes_id = tokenizer.encode(' Yes', add_special_tokens=False)[0]
                        _no_id = tokenizer.encode(' No', add_special_tokens=False)[0]
                    else:
                        _yes_id, _no_id = (8438, 2301)
                    yn_mask = (new_labels == _yes_id) | (new_labels == _no_id)
                    yn_pos = yn_mask.long().argmax(dim=1).clamp(min=1)
                    has_yn = yn_mask.any(dim=1)
                    first_ans = (new_labels != -100).float().argmax(dim=1).clamp(min=1)
                    yn_pos = torch.where(has_yn, yn_pos, first_ans)
                    last_prompt = (yn_pos - 1).clamp(min=0)
                    pooled = hidden[torch.arange(B_, device=hidden.device), last_prompt]
                    logits = cls_head(pooled.to(cls_head.weight.dtype)).squeeze(-1)
                    if is_main and step == 0 and (tokenizer is not None):
                        s = 0
                        yn = yn_pos[s].item()
                        lp = last_prompt[s].item()
                        seq_len = new_labels.shape[1]

                        def _tok(tid):
                            if tid == -100:
                                return '<masked>'
                            return repr(tokenizer.decode([tid]))
                        pass
                        pass
                        nonmasked = [(i, new_labels[s, i].item()) for i in range(seq_len) if new_labels[s, i].item() != -100]
                        pass
                        for offset, name in [(-1, 'lp-1'), (0, 'last_prompt'), (1, 'yn_pos(Yes/No)'), (2, 'yn+1')]:
                            pos = lp + offset
                            if 0 <= pos < seq_len:
                                tid = new_labels[s, pos].item()
                                pass
                        pass
                        pass
                    valid = cls_labels != -1
                    if valid.any():
                        cls_loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], cls_labels[valid].float())
                        loss = loss + cls_alpha * cls_loss
                        cls_loss_val = cls_loss.item()
                        preds = (logits[valid].detach() > 0).long()
                        cls_acc_val = (preds == cls_labels[valid]).float().mean().item()
            if train:
                scaler.scale(loss / cfg.grad_accum).backward()
                if (step + 1) % cfg.grad_accum == 0:
                    new_lr = get_lr(global_step, cfg.warmup_steps, cfg.lr, total_steps)
                    for pg in optimiser.param_groups:
                        pg['lr'] = new_lr
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad()
                    global_step += 1
                    if is_main and global_step % log_interval == 0:
                        avg = total_loss / max(n_batches, 1)
                        new_lr_now = get_lr(global_step, cfg.warmup_steps, cfg.lr, total_steps)
                        if cls_head is not None:
                            n_yes = (cls_labels == 1).sum().item()
                            n_no = (cls_labels == 0).sum().item()
                            n_unk = (cls_labels == -1).sum().item()
                            cls_info = f'  cls_loss={cls_loss_val:.4f}  cls_acc={cls_acc_val:.3f}  labels=[{n_yes}×Yes/{n_no}×No/{n_unk}×unk]'
                        else:
                            cls_info = ''
                        pass
                        if tokenizer is not None:
                            lbl0 = new_labels[0]
                            ans_pos = (lbl0 != -100).nonzero(as_tuple=True)[0]
                            if ans_pos.numel() > 0:
                                gt_ids = lbl0[ans_pos]
                                pred_ids = out.logits[0, ans_pos - 1, :].argmax(-1)
                                gt_txt = tokenizer.decode(gt_ids, skip_special_tokens=False)
                                pred_txt = tokenizer.decode(pred_ids, skip_special_tokens=False)
                                pass
                                pass
                    if is_main and ckpt_every > 0 and (global_step % ckpt_every == 0) and (ckpt_fn is not None):
                        ckpt_fn(global_step)
            total_loss += loss.item()
            n_batches += 1
    return (total_loss / max(n_batches, 1), global_step)

def train(cfg: TrainConfig) -> PrismMedGemma:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = rank == 0
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        cfg.device = f'cuda:{local_rank}'
        torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
    if is_main:
        pass
        pass
    trainval_mat, test_mat, feature_cols, cat_cols, cat_cards = build_ehr_matrix(cfg.train_val_csv, cfg.test_csv, cfg.patient_json)
    prior_ecg_idx = _extract_prior_ecg_index(cfg.patient_json)
    train_idx = LazyJSONIndex(cfg.train_json)
    val_idx = LazyJSONIndex(cfg.val_json)
    pass
    tokenizer = AutoTokenizer.from_pretrained(cfg.medgemma)
    tokenizer.add_tokens([DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN], special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model_args = PrismModelArgs(medgemma=cfg.medgemma, ecg_tower=cfg.ecg_tower, ecg_cfg=cfg.ecg_cfg, feature_cols=feature_cols, cat_feature_cols=cat_cols, cat_cardinalities=cat_cards, repr_dim=cfg.repr_dim, ehr_d_model=cfg.ehr_d_model, ehr_n_heads=cfg.ehr_n_heads, ehr_n_layers=cfg.ehr_n_layers, n_ecg_tokens=cfg.n_ecg_tokens, n_ehr_tokens=cfg.n_ehr_tokens, n_hist_tokens=cfg.n_hist_tokens, proj_hidden=cfg.proj_hidden)
    model = PrismMedGemma(model_args, tokenizer, bf16=cfg.bf16).to(cfg.device)
    model.print_trainable_params()
    if cfg.resume:
        PrismMedGemma.load_stage1_checkpoint(model, cfg.resume)
    ecg_cache = None
    if cfg.ecg_cache and os.path.isdir(cfg.ecg_cache):
        ecg_cache = ECGCache(cfg.ecg_cache)
    elif cfg.ecg_cache:
        pass
    collate_fn = CollateClass(h5_root=cfg.h5_root, tokenizer=tokenizer, max_seq_len=cfg.max_seq_len, ecg_cache=ecg_cache)
    ds_train = Stage1Dataset(train_idx, trainval_mat, prior_ecg_idx)
    ds_val = Stage1Dataset(val_idx, trainval_mat, prior_ecg_idx)
    if cfg.max_samples > 0 and len(ds_train) > cfg.max_samples:
        from torch.utils.data import Subset
        from llava.train.train_stage2 import build_or_load_strat_index, stratified_subset
        if is_main:
            strat_groups = build_or_load_strat_index(cfg.train_json)
            subset = stratified_subset(strat_groups, cfg.max_samples, seed=cfg.seed)
        else:
            subset = None
        if world_size > 1:
            if is_main:
                subset_t = torch.tensor(subset, dtype=torch.long)
            else:
                subset_t = torch.zeros(1, dtype=torch.long)
            length_t = torch.tensor([len(subset_t) if is_main else 0], dtype=torch.long).cuda()
            dist.broadcast(length_t, src=0)
            if not is_main:
                subset_t = torch.zeros(length_t[0].item(), dtype=torch.long)
            dist.broadcast(subset_t.cuda(), src=0)
            subset = subset_t.tolist()
        ds_train = Subset(ds_train, subset)
        if is_main:
            pass
    train_sampler = DistributedSampler(ds_train, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(ds_val, shuffle=False) if world_size > 1 else None
    _nw = cfg.num_workers
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=_nw, collate_fn=collate_fn, pin_memory=True, persistent_workers=_nw > 0, prefetch_factor=4 if _nw > 0 else None)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, sampler=val_sampler, num_workers=_nw, collate_fn=collate_fn, pin_memory=True, persistent_workers=_nw > 0, prefetch_factor=4 if _nw > 0 else None)
    try:
        model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
        if is_main:
            pass
    except Exception as e:
        if is_main:
            pass
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    d_llm = _get_base_model(model).language_model.config.hidden_size
    cls_head = nn.Linear(d_llm, 1, bias=True).to(cfg.device)
    nn.init.zeros_(cls_head.weight)
    nn.init.zeros_(cls_head.bias)
    pass
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(cls_head.parameters())
    optimiser = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(dl_train) * cfg.epochs // cfg.grad_accum
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.bf16 and cfg.device == 'cuda')
    optimiser.zero_grad()
    best_val = float('inf')
    global_step = 0
    best_path = os.path.join(cfg.output_dir, 'stage1_best.pt')
    base_model = _get_base_model(model)
    _prev_step_ckpt: list = []

    def _save_step_ckpt(step: int) -> None:
        path = os.path.join(cfg.output_dir, f'stage1_step{step}.pt')
        base_model.save_stage1_checkpoint(path, extra={'feature_cols': feature_cols, 'cat_feature_cols': cat_cols, 'cat_cardinalities': cat_cards, 'repr_dim': cfg.repr_dim, 'global_step': step, 'cls_head': cls_head.state_dict()})
        pass
        if _prev_step_ckpt and os.path.exists(_prev_step_ckpt[0]):
            os.remove(_prev_step_ckpt[0])
            pass
        _prev_step_ckpt[:] = [path]
    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, global_step = run_epoch(model, dl_train, optimiser, scaler, cfg, trainable, total_steps, global_step, train=True, log_interval=100, ckpt_every=10000, ckpt_fn=_save_step_ckpt if is_main else None, cls_head=cls_head, cls_alpha=cfg.cls_alpha, tokenizer=tokenizer)
        val_loss, _ = run_epoch(model, dl_val, None, None, cfg, trainable, total_steps, 0, train=False, cls_head=cls_head, cls_alpha=cfg.cls_alpha, tokenizer=tokenizer)
        if is_main:
            pass
            if val_loss < best_val:
                best_val = val_loss
                base_model.save_stage1_checkpoint(best_path, extra={'feature_cols': feature_cols, 'cat_feature_cols': cat_cols, 'cat_cardinalities': cat_cards, 'repr_dim': cfg.repr_dim, 'epoch': epoch, 'val_loss': val_loss, 'cls_head': cls_head.state_dict()})
                pass
    if is_main:
        pass
    if world_size > 1:
        dist.destroy_process_group()
    return base_model

@torch.no_grad()
def export_faiss_index(model: PrismMedGemma, trainval_mat: np.ndarray, test_mat: np.ndarray, feature_cols: List[str], df_tv_meta: pd.DataFrame, prior_ecg_idx: Dict[str, List[Dict]], cfg: TrainConfig, ecg_cache: Optional['ECGCache']=None) -> None:
    if not FAISS_OK:
        pass
        return
    out_dir = os.path.join(cfg.output_dir, 'faiss_ehr_index')
    os.makedirs(out_dir, exist_ok=True)
    raw = model
    if hasattr(raw, '_orig_mod'):
        raw = raw._orig_mod
    if hasattr(raw, 'module'):
        raw = raw.module
    model.eval()
    dev = cfg.device
    D_ehr = cfg.repr_dim
    D_ecg = getattr(getattr(raw, 'config', None), 'ecg_dim', None) or getattr(getattr(raw, 'ecg_tower', None), 'hidden_size', None) or 256
    TOP3 = 3

    def _embed_ehr(mat: np.ndarray, tag: str) -> np.ndarray:
        all_embs, bs = ([], 256)
        for s in tqdm(range(0, len(mat), bs), desc=f'  embed EHR {tag}'):
            chunk = torch.from_numpy(mat[s:s + bs]).to(dev)
            with torch.cuda.amp.autocast(enabled=cfg.bf16 and dev == 'cuda', dtype=torch.bfloat16):
                emb = model.ehr_tower(chunk).float()
            all_embs.append(emb.cpu().numpy())
        return np.vstack(all_embs)

    def _embed_one_ecg(ecg_path: str) -> Optional[np.ndarray]:
        if ecg_cache is not None:
            cached = ecg_cache.get(ecg_path)
            if cached is not None:
                return cached.astype(np.float32)
        wav = load_ecg_waveform(cfg.h5_root, ecg_path)
        if wav is None:
            return None
        t = torch.from_numpy(wav).unsqueeze(0).to(dev)
        with torch.cuda.amp.autocast(enabled=cfg.bf16 and dev == 'cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                emb = model.ecg_tower(t)
        return emb.float().squeeze(0).cpu().numpy()
    pass
    tv_embs = _embed_ehr(trainval_mat, 'trainval')
    test_embs = _embed_ehr(test_mat, 'test')
    np.save(os.path.join(out_dir, 'ehr_embeddings.npy'), tv_embs)
    np.save(os.path.join(out_dir, 'trainval_embeddings.npy'), tv_embs)
    np.save(os.path.join(out_dir, 'test_embeddings.npy'), test_embs)
    ecg_paths = df_tv_meta['general_file_name'].astype(str).tolist() if 'general_file_name' in df_tv_meta.columns else [''] * len(df_tv_meta)
    subj_ids = df_tv_meta['general_subject_id'].astype(str).tolist() if 'general_subject_id' in df_tv_meta.columns else [''] * len(df_tv_meta)
    meta = [{'row_idx': i, 'ecg_path': ep, 'subject_id': sid, 'has_history': ep in prior_ecg_idx} for i, (ep, sid) in enumerate(zip(ecg_paths, subj_ids))]
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    N_tv = len(tv_embs)
    pass
    d = tv_embs.shape[1]
    ehr_index = faiss.IndexFlatL2(d)
    ehr_index.add(tv_embs.astype(np.float32))
    topk_full = min(cfg.topk + 1, N_tv)
    _, topk_idx = ehr_index.search(tv_embs.astype(np.float32), topk_full)
    top3_ehr_idx = topk_idx[:, 1:TOP3 + 1]
    np.save(os.path.join(out_dir, 'topk_indices.npy'), topk_idx[:, 1:])
    faiss.write_index(ehr_index, os.path.join(out_dir, 'faiss.index'))
    retrieved_ehr_top3 = tv_embs[top3_ehr_idx]
    np.save(os.path.join(out_dir, 'retrieved_ehr_top3.npy'), retrieved_ehr_top3)
    pass
    pass
    ehr_row_to_hist_row = np.full(N_tv, -1, dtype=np.int32)
    hist_patients = []
    for m in meta:
        if m['has_history']:
            hist_patients.append((m['row_idx'], m['ecg_path'], prior_ecg_idx[m['ecg_path']]))
    max_H = max((len(recs) for _, _, recs in hist_patients), default=0)
    max_H = min(max_H, 10)
    N_hist = len(hist_patients)
    pass
    ecg_history_embs = np.zeros((N_hist, max_H, D_ecg), dtype=np.float32)
    ecg_history_mask = np.zeros((N_hist, max_H), dtype=bool)
    for h_row, (ehr_row, _, recs) in enumerate(tqdm(hist_patients, desc='  embed history ECGs')):
        recs_sorted = sorted(recs, key=lambda r: r['days_before'])
        for slot, rec in enumerate(recs_sorted[:max_H]):
            emb = _embed_one_ecg(rec['ecg_path'])
            if emb is not None:
                ecg_history_embs[h_row, slot] = emb
                ecg_history_mask[h_row, slot] = True
        ehr_row_to_hist_row[ehr_row] = h_row
    np.save(os.path.join(out_dir, 'ecg_history_embeddings.npy'), ecg_history_embs)
    np.save(os.path.join(out_dir, 'ecg_history_mask.npy'), ecg_history_mask)
    np.save(os.path.join(out_dir, 'ehr_row_to_hist_row.npy'), ehr_row_to_hist_row)
    pass
    pass
    cur_ecg_embs = np.zeros((N_tv, D_ecg), dtype=np.float32)
    cur_ecg_valid = np.zeros(N_tv, dtype=bool)
    for m in tqdm(meta, desc='  embed current ECGs'):
        emb = _embed_one_ecg(m['ecg_path'])
        if emb is not None:
            cur_ecg_embs[m['row_idx']] = emb
            cur_ecg_valid[m['row_idx']] = True
    np.save(os.path.join(out_dir, 'current_ecg_embeddings.npy'), cur_ecg_embs)
    ecg_index = faiss.IndexFlatL2(D_ecg)
    ecg_index.add(cur_ecg_embs.astype(np.float32))
    pass
    _, ecg_topk_idx = ecg_index.search(cur_ecg_embs.astype(np.float32), TOP3 + 1)
    retrieved_ecg_top3 = np.zeros((N_tv, TOP3, D_ecg), dtype=np.float32)
    for m in tqdm(meta, desc='  build retrieved_ecg_top3'):
        r = m['row_idx']
        h_row = int(ehr_row_to_hist_row[r])
        if h_row >= 0:
            valid_slots = np.where(ecg_history_mask[h_row])[0][:TOP3]
            for k, slot in enumerate(valid_slots):
                retrieved_ecg_top3[r, k] = ecg_history_embs[h_row, slot]
        else:
            neighbors = [idx for idx in ecg_topk_idx[r] if idx != r][:TOP3]
            for k, nb in enumerate(neighbors):
                retrieved_ecg_top3[r, k] = cur_ecg_embs[nb]
    np.save(os.path.join(out_dir, 'retrieved_ecg_top3.npy'), retrieved_ecg_top3)
    pass
    pass
    pass
    pass
    pass
    pass
    pass
    pass

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument('--train_json', default='/projects/prjs1786/CNDP/New_Code/train_raft1.json')
    p.add_argument('--val_json', default='/projects/prjs1786/CNDP/New_Code/test_raft1.json')
    p.add_argument('--train_val_csv', required=True)
    p.add_argument('--test_csv', required=True)
    p.add_argument('--patient_json', default='')
    p.add_argument('--h5_root', default='')
    p.add_argument('--ecg_cache', default='')
    p.add_argument('--output_dir', default='./stage1_out')
    p.add_argument('--resume', default='')
    p.add_argument('--medgemma', default='google/medgemma-4b-it')
    p.add_argument('--ecg_tower', default=PrismModelArgs.ecg_tower)
    p.add_argument('--ecg_cfg', default=PrismModelArgs.ecg_cfg)
    p.add_argument('--repr_dim', type=int, default=512)
    p.add_argument('--ehr_d_model', type=int, default=192)
    p.add_argument('--ehr_n_heads', type=int, default=4)
    p.add_argument('--ehr_n_layers', type=int, default=3)
    p.add_argument('--n_ecg_tokens', type=int, default=8)
    p.add_argument('--n_ehr_tokens', type=int, default=8)
    p.add_argument('--n_hist_tokens', type=int, default=8)
    p.add_argument('--proj_hidden', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--grad_accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.0002)
    p.add_argument('--warmup_steps', type=int, default=100, help='LR warmup steps (default 100)')
    p.add_argument('--max_samples', type=int, default=0, help='Cap training set size (0=full). Stratified by task×label.')
    p.add_argument('--max_seq_len', type=int, default=512)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--cls_alpha', type=float, default=0.1, help='Weight of binary cls loss (0 = disable cls head)')
    p.add_argument('--no_export', action='store_true')
    p.add_argument('--no_bf16', action='store_true')
    a = p.parse_args()
    cfg = TrainConfig()
    for k, v in vars(a).items():
        if k == 'no_export':
            cfg.export_after_train = not v
        elif k == 'no_bf16':
            cfg.bf16 = not v
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def main():
    cfg = parse_args()
    model = train(cfg)
    is_main = int(os.environ.get('RANK', '0')) == 0
    if cfg.export_after_train and is_main:
        tv_mat, test_mat, feat_cols, cat_cols, cat_cards = build_ehr_matrix(cfg.train_val_csv, cfg.test_csv, cfg.patient_json)
        prior_ecg_idx = _extract_prior_ecg_index(cfg.patient_json)
        df_tv = pd.read_csv(cfg.train_val_csv, low_memory=False)
        ecg_cache = None
        if cfg.ecg_cache and os.path.isdir(cfg.ecg_cache):
            try:
                ecg_cache = ECGCache(cfg.ecg_cache)
            except Exception as e:
                pass
        export_faiss_index(model, tv_mat, test_mat, feat_cols, df_tv, prior_ecg_idx, cfg, ecg_cache=ecg_cache)
if __name__ == '__main__':
    main()

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm
_LLAVA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _LLAVA_ROOT not in sys.path:
    sys.path.insert(0, _LLAVA_ROOT)
from llava.constants import DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, IGNORE_INDEX
from llava.model.language_model.prism_medgemma import PrismMedGemma, PrismModelArgs
from llava.model.multimodal_projector.builder import build_soft_token_projector
from llava.train.train_stage1 import LazyJSONIndex, ECGCache, build_ehr_matrix, _extract_prior_ecg_index
from llava.train.train_stage2 import DEFAULT_RET_EHR_TOKEN, DEFAULT_RET_ECG_TOKEN, RetDataset, RetCollateClass, prepare_inputs_5tokens
from transformers import AutoTokenizer
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    pass
    sys.exit(1)

def _task_type(sample_id: str) -> str:
    return re.sub('_\\d+$', '', sample_id)

def _group(sample_id: str) -> str:
    if sample_id.startswith('deterioration_mortality'):
        return 'mortality'
    elif sample_id.startswith('deterioration_icu'):
        return 'icu'
    elif sample_id.startswith('deterioration_'):
        return 'other_det'
    elif sample_id.startswith('diagnoses_'):
        return 'diagnoses'
    else:
        return 'other'

def _auc_from_lists(labels: List[int], scores: List[float]) -> Optional[float]:
    if len(labels) < 2:
        return None
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = auc = prev_fp = prev_tp = 0
    for _, lbl in pairs:
        if lbl == 1:
            tp += 1
        else:
            fp += 1
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_fp, prev_tp = (fp, tp)
    return auc / (pos * neg)

def _acc_from_lists(labels: List[int], preds: List[int]) -> float:
    if not labels:
        return float('nan')
    return sum((l == p for l, p in zip(labels, preds))) / len(labels)

def _fmt_auc(v: Optional[float]) -> str:
    return f'{v:.4f}' if v is not None else '  N/A '

def print_group_results(group_name: str, task_results: Dict[str, dict], print_per_task: bool=True) -> Tuple[Optional[float], Optional[float], float, float]:
    sep = '─' * 88
    pass
    pass
    pass
    pass
    pass
    cls_aucs, llm_aucs, cls_accs, llm_accs, n_samples = ([], [], [], [], [])
    for task, data in sorted(task_results.items()):
        labels = data['labels']
        cls_scores = data['cls_scores']
        llm_scores = data['llm_scores']
        llm_preds = data['llm_preds']
        n = len(labels)
        n_samples.append(n)
        cls_auc = _auc_from_lists(labels, cls_scores)
        if cls_auc is not None:
            cls_aucs.append(cls_auc)
        llm_auc = _auc_from_lists(labels, llm_scores)
        llm_acc = _acc_from_lists(labels, llm_preds)
        if llm_auc is not None:
            llm_aucs.append(llm_auc)
        llm_accs.append(llm_acc)
        cls_acc = _acc_from_lists(labels, [1 if s >= 0.5 else 0 for s in cls_scores])
        cls_accs.append(cls_acc)
        if print_per_task:
            llm_acc_str = f'{llm_acc:.4f}' if not np.isnan(llm_acc) else '   nan'
            pass
    pass
    mean_cls_auc = float(np.mean(cls_aucs)) if cls_aucs else None
    mean_llm_auc = float(np.mean(llm_aucs)) if llm_aucs else None
    mean_cls_acc = float(np.mean(cls_accs)) if cls_accs else float('nan')
    mean_llm_acc = float(np.mean(llm_accs)) if llm_accs else float('nan')
    total_n = sum(n_samples)
    n_cls_above = sum((1 for a in cls_aucs if a > 0.8))
    n_llm_above = sum((1 for a in llm_aucs if a > 0.8))
    llm_acc_str = f'{mean_llm_acc:.4f}' if not np.isnan(mean_llm_acc) else '   nan'
    pass
    pass
    return (mean_cls_auc, mean_llm_auc, mean_cls_acc, mean_llm_acc)

class LengthSortedSampler(Sampler):

    def __init__(self, index: 'LazyJSONIndex', batch_size: int) -> None:
        lengths = [len(index[i].get('prompt', '')) for i in range(len(index))]
        self._indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)

class EvalRetDataset(RetDataset):

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item['sample_id'] = self.index[idx]['id']
        item['orig_idx'] = idx
        return item

class EvalRetCollate(RetCollateClass):

    def __call__(self, batch: List[dict]) -> dict:
        result = super().__call__(batch)
        result['sample_ids'] = [item['sample_id'] for item in batch]
        result['orig_idxs'] = [item['orig_idx'] for item in batch]
        return result

def load_stage2_ret_checkpoint(ckpt_dir, model, ret_ehr_projector, ret_ecg_projector, device):
    s1_path = os.path.join(ckpt_dir, 'stage1_weights.pt')
    pass
    s1 = torch.load(s1_path, map_location='cpu', weights_only=False)
    feature_cols = s1.get('feature_cols', [])
    cat_cols = s1.get('cat_feature_cols', [])
    cat_cards = s1.get('cat_cardinalities', [])
    repr_dim = s1.get('repr_dim', 512)
    cls_head_sd = s1.get('cls_head', None)
    step = s1.get('step', 0)
    pass
    model_sd = model.state_dict()
    update_keys = {k: v for k, v in s1.items() if isinstance(v, torch.Tensor) and k in model_sd}
    missing = [k for k in model_sd if k not in s1 and (not k.startswith('language_model.'))]
    unexpected = [k for k in s1 if isinstance(s1[k], torch.Tensor) and k not in model_sd]
    model_sd.update(update_keys)
    model.load_state_dict(model_sd, strict=False)
    pass
    ret_ehr_sd = s1.get('ret_ehr_projector', None)
    ret_ecg_sd = s1.get('ret_ecg_projector', None)
    if ret_ehr_sd is not None:
        ret_ehr_projector.load_state_dict(ret_ehr_sd)
        pass
    else:
        pass
    if ret_ecg_sd is not None:
        ret_ecg_projector.load_state_dict(ret_ecg_sd)
        pass
    else:
        pass
    pass
    model.language_model = PeftModel.from_pretrained(model.language_model, ckpt_dir, is_trainable=False)
    model.language_model = model.language_model.merge_and_unload()
    pass
    model.to(device)
    ret_ehr_projector.to(device)
    ret_ecg_projector.to(device)
    model.eval()
    ret_ehr_projector.eval()
    ret_ecg_projector.eval()
    return (cls_head_sd, feature_cols, cat_cols, cat_cards, repr_dim)

def evaluate(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_cudnn_sdp(False)
    ckpt_dir = args.checkpoint
    assert os.path.isdir(ckpt_dir), f'Checkpoint directory not found: {ckpt_dir}'
    pass
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.add_tokens([DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, DEFAULT_RET_EHR_TOKEN, DEFAULT_RET_ECG_TOKEN], special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    ret_ehr_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_RET_EHR_TOKEN)
    ret_ecg_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_RET_ECG_TOKEN)
    pass
    s1_path = os.path.join(ckpt_dir, 'stage1_weights.pt')
    _s1_meta = torch.load(s1_path, map_location='cpu', weights_only=False)
    ckpt_feature_cols = _s1_meta.get('feature_cols', [])
    ckpt_cat_cols = _s1_meta.get('cat_feature_cols', [])
    ckpt_cat_cards = _s1_meta.get('cat_cardinalities', [])
    ckpt_repr_dim = _s1_meta.get('repr_dim', 512)
    del _s1_meta
    _, test_mat, fc, cc, ccard = build_ehr_matrix(args.train_val_csv, args.test_csv, args.patient_json)
    if ckpt_feature_cols:
        fc_map = {c: i for i, c in enumerate(fc)}
        try:
            col_idx = [fc_map[c] for c in ckpt_feature_cols]
            test_mat = test_mat[:, col_idx]
            pass
        except KeyError as e:
            pass
            ckpt_feature_cols, ckpt_cat_cols, ckpt_cat_cards = (fc, cc, ccard)
    else:
        ckpt_feature_cols, ckpt_cat_cols, ckpt_cat_cards = (fc, cc, ccard)
    prior_ecg_idx = _extract_prior_ecg_index(args.patient_json)
    pass
    ret_ehr_path = os.path.join(args.faiss_dir, 'retrieved_ehr_top3.npy')
    ret_ecg_path = os.path.join(args.faiss_dir, 'retrieved_ecg_top3.npy')
    if not os.path.isfile(ret_ehr_path):
        raise FileNotFoundError(f'retrieved_ehr_top3.npy not found in {args.faiss_dir}')
    if not os.path.isfile(ret_ecg_path):
        raise FileNotFoundError(f'retrieved_ecg_top3.npy not found in {args.faiss_dir}')
    ret_ehr_embs = np.load(ret_ehr_path)
    ret_ecg_embs = np.load(ret_ecg_path)
    pass
    ret_ehr_dim = ret_ehr_embs.shape[-1]
    ret_ecg_dim = ret_ecg_embs.shape[-1]
    do_faiss_insp = False
    topk_indices_all = None
    faiss_meta = None
    subj_outcomes = {}
    outcome_cols: List[str] = []
    actual_k = 0
    faiss_insp_acc: Dict[str, Dict[int, Dict[str, dict]]] = {}
    if args.analysis_k > 0:
        _topk_path = os.path.join(args.faiss_dir, 'topk_indices.npy')
        _meta_path = os.path.join(args.faiss_dir, 'meta.json')
        if os.path.isfile(_topk_path) and os.path.isfile(_meta_path):
            topk_indices_all = np.load(_topk_path)
            with open(_meta_path) as _f:
                faiss_meta = json.load(_f)
            actual_k = min(args.analysis_k, topk_indices_all.shape[1])
            try:
                import pandas as _pd
                _tv = _pd.read_csv(args.train_val_csv)
                if 'subject_id' in _tv.columns:
                    _tv['subject_id'] = _tv['subject_id'].astype(str)
                    subj_outcomes = _tv.set_index('subject_id').to_dict(orient='index')
                outcome_cols = [c for c in _tv.columns if c != 'subject_id' and set(_tv[c].dropna().unique()).issubset({0, 1, 0.0, 1.0}) and (_tv[c].notna().sum() >= 100)]
                del _tv
            except Exception as _e:
                pass
            do_faiss_insp = True
            pass
            pass
        else:
            pass

    def _faiss_acc(task: str, gt: int, col: str) -> dict:
        if task not in faiss_insp_acc:
            faiss_insp_acc[task] = {0: {}, 1: {}}
        if col not in faiss_insp_acc[task][gt]:
            faiss_insp_acc[task][gt][col] = {'sum': 0.0, 'count': 0}
        return faiss_insp_acc[task][gt][col]
    pass
    model_args = PrismModelArgs(medgemma=args.medgemma, ecg_tower=args.ecg_tower, ecg_cfg=args.ecg_cfg, feature_cols=ckpt_feature_cols, cat_feature_cols=ckpt_cat_cols, cat_cardinalities=ckpt_cat_cards, repr_dim=ckpt_repr_dim, ehr_d_model=args.ehr_d_model, ehr_n_heads=args.ehr_n_heads, ehr_n_layers=args.ehr_n_layers, n_ecg_tokens=args.n_ecg_tokens, n_ehr_tokens=args.n_ehr_tokens, n_hist_tokens=args.n_hist_tokens, proj_hidden=args.proj_hidden)
    model = PrismMedGemma(model_args, tokenizer, bf16=True)
    d_llm = model.language_model.config.hidden_size
    ret_ehr_projector = build_soft_token_projector(ret_ehr_dim, d_llm, args.n_ret_ehr_tokens, args.proj_hidden).to(torch.bfloat16)
    ret_ecg_projector = build_soft_token_projector(ret_ecg_dim, d_llm, args.n_ret_ecg_tokens, args.proj_hidden).to(torch.bfloat16)
    pass
    pass
    cls_head_sd, *_ = load_stage2_ret_checkpoint(ckpt_dir, model, ret_ehr_projector, ret_ecg_projector, device)
    cls_head = nn.Linear(d_llm, 1, bias=True).to(device)
    if cls_head_sd is not None:
        cls_head.load_state_dict(cls_head_sd)
        pass
    else:
        pass
    cls_head.eval()
    ecg_cache = None
    if args.ecg_cache and os.path.isdir(args.ecg_cache):
        ecg_cache = ECGCache(args.ecg_cache)
    test_idx = LazyJSONIndex(args.test_json)
    ds_test = EvalRetDataset(test_idx, test_mat, prior_ecg_idx)
    collate = EvalRetCollate(h5_root=args.h5_root, tokenizer=tokenizer, max_seq_len=args.max_seq_len, ret_ehr_embs=ret_ehr_embs, ret_ecg_embs=ret_ecg_embs, ecg_cache=ecg_cache)
    if args.sort_by_length:
        pass
        _sampler = LengthSortedSampler(test_idx, args.batch_size)
    else:
        _sampler = None
    dl = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, sampler=_sampler, num_workers=args.num_workers, collate_fn=collate, pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)
    pass

    def _all_tok_ids(word: str) -> List[int]:
        vocab = tokenizer.get_vocab()
        candidates = [word, word.capitalize(), f'▁{word}', f'▁{word.capitalize()}']
        ids = []
        for c in candidates:
            tid = vocab.get(c, None)
            if tid is not None and tid not in ids:
                ids.append(tid)
        if not ids:
            tids = tokenizer(word, add_special_tokens=False)['input_ids']
            if tids:
                ids.append(tids[-1])
        return ids
    yes_ids = _all_tok_ids('yes')
    no_ids = _all_tok_ids('no')
    pass
    pass
    task_data: Dict[str, dict] = defaultdict(lambda: {'labels': [], 'cls_scores': [], 'llm_scores': [], 'llm_preds': []})
    autocast_ctx = torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16)
    do_feat_imp = hasattr(model, 'ehr_tower') and model.ehr_tower.cont_weight is not None and (args.analysis_k >= 0)
    feat_imp_acc: Dict[str, dict] = {}
    cont_names: List[str] = []
    cont_weight_norms_cpu: Optional[torch.Tensor] = None
    if do_feat_imp:
        cont_names = [model.ehr_tower.feature_cols[i] for i in model.ehr_tower.cont_indices]
        cont_weight_norms_cpu = model.ehr_tower.cont_weight.detach().float().norm(dim=1).cpu()
        pass

    def _feat_acc(task: str) -> dict:
        if task not in feat_imp_acc:
            feat_imp_acc[task] = {'sum': np.zeros(len(cont_names), dtype=np.float64), 'count': 0}
        return feat_imp_acc[task]
    n_total = 0
    batch_idx = 0
    all_records = []
    jsonl_path = csv_path = ''
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        jsonl_path = os.path.join(args.output_dir, 'eval_prompt_predictions.jsonl')
        csv_path = os.path.join(args.output_dir, 'eval_prompt_predictions.csv')
        pass
        pass
    csv_fieldnames = ['id', 'task', 'group', 'gt', 'gt_text', 'cls_score', 'cls_pred', 'llm_score', 'llm_pred']
    special_ids = {model.ecg_token_id, model.ehr_token_id, model.hist_token_id, ret_ehr_token_id, ret_ecg_token_id}
    with open(jsonl_path, 'w') if jsonl_path else open(os.devnull, 'w') as jsonl_fh, torch.no_grad(), autocast_ctx:
        for batch in tqdm(dl, desc='Evaluating (prompt only)'):
            ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            lbs = batch['labels'].to(device)
            ehr = batch['ehr_feat'].to(device)
            ret_ehr = batch['ret_ehr'].to(device)
            ret_ecg = batch['ret_ecg'].to(device)
            sample_ids = batch['sample_ids']
            orig_idxs = batch['orig_idxs']
            B = len(sample_ids)
            gt_labels = batch['cls_labels'].numpy()
            if batch['ecg_embeddings'] is not None:
                ecg_waves_or_emb = batch['ecg_embeddings'].to(device)
                use_cached_ecg = True
            else:
                ecg_waves_or_emb = batch['ecg_waveforms'].to(device)
                use_cached_ecg = False
            ecg_emb_list: List[Optional[torch.Tensor]] = []
            days_l: List[Optional[torch.Tensor]] = []
            for pre_emb, pw, days in zip(batch['prior_ecg_embs_list'], batch['prior_ecg_waves_list'], batch['days_list']):
                if pre_emb is not None:
                    ecg_emb_list.append(pre_emb.to(device))
                elif pw is not None:
                    ecg_emb_list.append(model.ecg_tower(pw.to(device)))
                else:
                    ecg_emb_list.append(None)
                days_l.append(days.to(device) if days is not None else None)
            if use_cached_ecg:
                ecg_soft = model.ecg_projector(ecg_waves_or_emb.to(model.ecg_projector.net[0].weight.dtype))
            else:
                ecg_feat = model.ecg_tower(ecg_waves_or_emb)
                ecg_soft = model.ecg_projector(ecg_feat.to(model.ecg_projector.net[0].weight.dtype))
            ehr_soft = model.encode_ehr(ehr)
            hist_soft = model.encode_ecg_history(ecg_emb_list, days_l)
            proj_dtype = ret_ehr_projector.net[0].weight.dtype
            ret_ehr_soft = ret_ehr_projector(ret_ehr.to(proj_dtype))
            ret_ecg_soft = ret_ecg_projector(ret_ecg.to(proj_dtype))
            tok_map = {model.ecg_token_id: ecg_soft, model.ehr_token_id: ehr_soft, model.hist_token_id: hist_soft, ret_ehr_token_id: ret_ehr_soft, ret_ecg_token_id: ret_ecg_soft}
            embed_fn = model.get_embed_tokens()
            new_embeds, new_labels, new_attn = prepare_inputs_5tokens(ids, attn, lbs, tok_map, embed_fn)
            prompt_lens = []
            for b in range(B):
                non_ign = (new_labels[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                if len(non_ign) > 0:
                    prompt_lens.append(non_ign[0].item())
                else:
                    prompt_lens.append(new_labels.size(1))
            max_p = max(prompt_lens)
            prompt_embeds = new_embeds[:, :max_p, :].clone()
            prompt_attn = new_attn[:, :max_p].clone()
            for b in range(B):
                if prompt_lens[b] < max_p:
                    prompt_embeds[b, prompt_lens[b]:] = 0
                    prompt_attn[b, prompt_lens[b]:] = 0
            out = model.language_model(inputs_embeds=prompt_embeds, attention_mask=prompt_attn, output_hidden_states=True)
            pred_pos = torch.tensor([max(prompt_lens[b] - 1, 0) for b in range(B)], device=device, dtype=torch.long)
            pp_exp = pred_pos.view(B, 1, 1).expand(B, 1, out.logits.shape[-1])
            last_logits = out.logits.gather(1, pp_exp).squeeze(1).float()
            yes_logits = last_logits[:, yes_ids].max(dim=1).values
            no_logits = last_logits[:, no_ids].max(dim=1).values
            yn_logits = torch.stack([yes_logits, no_logits], dim=1)
            llm_scores = torch.softmax(yn_logits, dim=1)[:, 0]
            llm_preds = (llm_scores >= 0.5).long()
            hidden = out.hidden_states[-1][:, :max_p, :]
            pool_mask = prompt_attn.float().unsqueeze(-1)
            pooled = (hidden * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
            cls_logit = cls_head(pooled.to(cls_head.weight.dtype)).squeeze(-1)
            cls_score = torch.sigmoid(cls_logit)
            if do_feat_imp:
                x_cont = ehr[:, model.ehr_tower.cont_indices].float().cpu()
                imp = (x_cont.abs() * cont_weight_norms_cpu).numpy()
                for b in range(B):
                    _acc = _feat_acc(_task_type(sample_ids[b]))
                    _acc['sum'] += imp[b]
                    _acc['count'] += 1
            if do_faiss_insp and outcome_cols:
                for b in range(B):
                    oidx = orig_idxs[b]
                    task_b = _task_type(sample_ids[b])
                    gt_b = int(gt_labels[b])
                    if oidx >= len(topk_indices_all):
                        continue
                    for nidx in topk_indices_all[oidx][:actual_k]:
                        if nidx < 0 or nidx >= len(faiss_meta):
                            continue
                        subj = str(faiss_meta[nidx].get('subject_id', ''))
                        if subj not in subj_outcomes:
                            continue
                        row = subj_outcomes[subj]
                        for col in outcome_cols:
                            val = row.get(col, None)
                            if val is not None:
                                try:
                                    fval = float(val)
                                    if not np.isnan(fval):
                                        _r = _faiss_acc(task_b, gt_b, col)
                                        _r['sum'] += fval
                                        _r['count'] += 1
                                except (TypeError, ValueError):
                                    pass
            cls_score_np = cls_score.cpu().float().numpy()
            llm_score_np = llm_scores.cpu().float().numpy()
            llm_pred_np = llm_preds.cpu().numpy()
            lbs_cpu = lbs.cpu()
            ids_cpu = ids.cpu()
            for b in range(B):
                sid = sample_ids[b]
                gt_lbl = int(gt_labels[b])
                cls_s = float(cls_score_np[b])
                llm_s = float(llm_score_np[b])
                llm_p = int(llm_pred_np[b])
                cls_p = 1 if cls_s >= 0.5 else 0
                task = _task_type(sid)
                ans_mask = lbs_cpu[b] != IGNORE_INDEX
                gt_text = tokenizer.decode(ids_cpu[b][ans_mask], skip_special_tokens=True).strip()
                task_data[task]['labels'].append(gt_lbl)
                task_data[task]['cls_scores'].append(cls_s)
                task_data[task]['llm_scores'].append(llm_s)
                task_data[task]['llm_preds'].append(llm_p)
                record = {'id': sid, 'task': task, 'group': _group(sid), 'gt': gt_lbl, 'gt_text': gt_text, 'cls_score': round(cls_s, 5), 'cls_pred': cls_p, 'llm_score': round(llm_s, 5), 'llm_pred': llm_p}
                jsonl_fh.write(json.dumps(record) + '\n')
                all_records.append(record)
                n_total += 1
            batch_idx += 1
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                gt_str = 'Yes' if gt_labels[0] == 1 else 'No'
                cls_str = 'Yes' if cls_score_np[0] >= 0.5 else 'No'
                llm_str = 'Yes' if llm_pred_np[0] == 1 else 'No'
                prompt_ids = ids_cpu[0][lbs_cpu[0] == IGNORE_INDEX]
                prompt_ids = prompt_ids[~torch.isin(prompt_ids, torch.tensor(list(special_ids), dtype=torch.long))]
                prompt_txt = tokenizer.decode(prompt_ids, skip_special_tokens=True).strip()
                prompt_short = prompt_txt[-300:] if len(prompt_txt) > 300 else prompt_txt
                ans_mask0 = lbs_cpu[0] != IGNORE_INDEX
                gt_text0 = tokenizer.decode(ids_cpu[0][ans_mask0], skip_special_tokens=True).strip()
                sep = '  ' + '─' * 84
                pass
                pass
                pass
                pass
                pass
                pass
                pass
                pass
    if do_feat_imp and args.output_dir and feat_imp_acc:
        feat_imp_out = {}
        for task, acc in feat_imp_acc.items():
            if acc['count'] == 0:
                continue
            mean_imp = acc['sum'] / max(acc['count'], 1)
            top_features = sorted([(cont_names[i], float(mean_imp[i])) for i in range(len(cont_names)) if mean_imp[i] > 1e-09], key=lambda x: -x[1])[:20]
            feat_imp_out[task] = {'n_samples': acc['count'], 'top20': top_features}
        fi_path = os.path.join(args.output_dir, 'feat_importance_by_task.json')
        with open(fi_path, 'w') as f:
            json.dump(feat_imp_out, f, indent=2)
        pass
    if do_faiss_insp and args.output_dir and faiss_insp_acc:
        faiss_out: dict = {}
        for task, gt_dict in faiss_insp_acc.items():
            task_out: dict = {}
            for gt_label, col_dict in gt_dict.items():
                col_rates: dict = {}
                for col, acc in col_dict.items():
                    if acc['count'] > 0:
                        col_rates[col] = {'mean': round(acc['sum'] / acc['count'], 4), 'count': acc['count']}
                if col_rates:
                    task_out[f'gt_{gt_label}'] = col_rates
            if task_out:
                faiss_out[task] = task_out
        faiss_path = os.path.join(args.output_dir, 'faiss_retrieval_stats.json')
        with open(faiss_path, 'w') as f:
            json.dump(faiss_out, f, indent=2)
        pass
    if csv_path and all_records:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(all_records)
        pass
    if jsonl_path:
        pass
    pass
    mort_tasks = {t: d for t, d in task_data.items() if t.startswith('deterioration_mortality')}
    icu_tasks = {t: d for t, d in task_data.items() if t.startswith('deterioration_icu')}
    other_tasks = {t: d for t, d in task_data.items() if t.startswith('deterioration_') and (not t.startswith('deterioration_mortality')) and (not t.startswith('deterioration_icu'))}
    diag_tasks = {t: d for t, d in task_data.items() if t.startswith('diagnoses_')}
    group_cls_aucs, group_llm_aucs = ([], [])
    group_cls_accs, group_llm_accs = ([], [])
    for g_name, g_tasks, per_task in [('Mortality (deterioration_mortality_*)', mort_tasks, True), ('ICU (deterioration_icu_*)', icu_tasks, True), ('Other Deterioration (deterioration_*)', other_tasks, True), ('Diagnoses (diagnoses_*)', diag_tasks, False)]:
        if not g_tasks:
            continue
        c_auc, l_auc, c_acc, l_acc = print_group_results(g_name, g_tasks, per_task)
        if c_auc is not None:
            group_cls_aucs.append(c_auc)
        if l_auc is not None:
            group_llm_aucs.append(l_auc)
        if not np.isnan(c_acc):
            group_cls_accs.append(c_acc)
        if not np.isnan(l_acc):
            group_llm_accs.append(l_acc)
    pass
    pass
    pass
    mean_c_auc = float(np.mean(group_cls_aucs)) if group_cls_aucs else None
    mean_l_auc = float(np.mean(group_llm_aucs)) if group_llm_aucs else None
    mean_c_acc = float(np.mean(group_cls_accs)) if group_cls_accs else float('nan')
    mean_l_acc = float(np.mean(group_llm_accs)) if group_llm_accs else float('nan')
    pass
    pass
    pass
    if args.output_dir:
        summary = {'checkpoint': ckpt_dir, 'faiss_dir': args.faiss_dir, 'n_samples': n_total, 'cls_mean_auc': mean_c_auc, 'llm_mean_auc': mean_l_auc, 'cls_mean_acc': mean_c_acc, 'llm_mean_acc': mean_l_acc, 'groups': {'mortality': {'cls_auc': group_cls_aucs[0] if len(group_cls_aucs) > 0 else None, 'llm_auc': group_llm_aucs[0] if len(group_llm_aucs) > 0 else None}, 'icu': {'cls_auc': group_cls_aucs[1] if len(group_cls_aucs) > 1 else None, 'llm_auc': group_llm_aucs[1] if len(group_llm_aucs) > 1 else None}, 'other_det': {'cls_auc': group_cls_aucs[2] if len(group_cls_aucs) > 2 else None, 'llm_auc': group_llm_aucs[2] if len(group_llm_aucs) > 2 else None}, 'diagnoses': {'cls_auc': group_cls_aucs[3] if len(group_cls_aucs) > 3 else None, 'llm_auc': group_llm_aucs[3] if len(group_llm_aucs) > 3 else None}}}
        summ_path = os.path.join(args.output_dir, 'eval_prompt_summary.json')
        with open(summ_path, 'w') as f:
            json.dump(summary, f, indent=2)
        pass

def parse_args():
    p = argparse.ArgumentParser(description='PRISM Stage 2 Retrieval Prompt-Only Evaluation')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--faiss_dir', required=True)
    p.add_argument('--test_json', default='/projects/prjs1786/CNDP/New_Code/test_raft2.json')
    p.add_argument('--train_val_csv', required=True)
    p.add_argument('--test_csv', required=True)
    p.add_argument('--patient_json', required=True)
    p.add_argument('--h5_root', required=True)
    p.add_argument('--ecg_cache', default='')
    p.add_argument('--output_dir', default='')
    p.add_argument('--medgemma', default='google/medgemma-4b-it')
    p.add_argument('--ecg_tower', default='/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder/models/best.pt')
    p.add_argument('--ecg_cfg', default='/projects/prjs1786/mimic_data/LLARVA/llava/model/ecg_encoder/configs/config0.json')
    p.add_argument('--ehr_d_model', type=int, default=192)
    p.add_argument('--ehr_n_heads', type=int, default=4)
    p.add_argument('--ehr_n_layers', type=int, default=3)
    p.add_argument('--n_ecg_tokens', type=int, default=8)
    p.add_argument('--n_ehr_tokens', type=int, default=8)
    p.add_argument('--n_hist_tokens', type=int, default=8)
    p.add_argument('--n_ret_ehr_tokens', type=int, default=8)
    p.add_argument('--n_ret_ecg_tokens', type=int, default=8)
    p.add_argument('--proj_hidden', type=int, default=1024)
    p.add_argument('--max_seq_len', type=int, default=2048)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--sort_by_length', action='store_true', help='Sort by prompt length to minimise padding waste (~30-40%% faster)')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--analysis_k', type=int, default=10, help='Top-K FAISS neighbors to inspect per test sample (0 = disable analyses)')
    return p.parse_args()
if __name__ == '__main__':
    evaluate(parse_args())

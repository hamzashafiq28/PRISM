from __future__ import annotations
import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x
_LLAVA_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _LLAVA_ROOT not in sys.path:
    sys.path.insert(0, _LLAVA_ROOT)
from llava.constants import DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, IGNORE_INDEX
from llava.model.language_model.prism_medgemma import PrismMedGemma, PrismModelArgs
from llava.model.multimodal_projector.builder import build_soft_token_projector
from llava.train.train_stage1 import LazyJSONIndex, Stage1Dataset, CollateClass, ECGCache, build_ehr_matrix, _extract_prior_ecg_index, _ENGLISH_PREFIX, get_lr
from llava.train.train_stage2 import apply_lora_to_language_model, freeze_stage1_components, build_or_load_strat_index, stratified_subset, Stage2Config
from transformers import AutoTokenizer
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    pass
    sys.exit(1)
DEFAULT_RET_EHR_TOKEN = '<retrived_ehr>'
DEFAULT_RET_ECG_TOKEN = '<retrived_ecg_history>'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_cudnn_sdp(False)

@dataclass
class Stage2RetConfig(Stage2Config):
    faiss_dir: str = ''
    n_ret_ehr_tokens: int = 8
    n_ret_ecg_tokens: int = 8
    resume_from: str = ''
    val_max_samples: int = 0

class RetDataset(Stage1Dataset):

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        entry = self.index[idx]
        ehr_row = entry['faiss_npy']['ehr_row']
        human_text = entry['conversations'][0]['value']
        item['ehr_row'] = int(ehr_row) if ehr_row is not None else -1
        item['prompt'] = _ENGLISH_PREFIX + human_text
        return item

class RetCollateClass(CollateClass):

    def __init__(self, h5_root: str, tokenizer, max_seq_len: int, ret_ehr_embs: np.ndarray, ret_ecg_embs: np.ndarray, ecg_cache: Optional[ECGCache]=None):
        super().__init__(h5_root, tokenizer, max_seq_len, ecg_cache)
        self.ret_ehr_embs = ret_ehr_embs
        self.ret_ecg_embs = ret_ecg_embs
        self._ehr_dim = ret_ehr_embs.shape[-1]
        self._ecg_dim = ret_ecg_embs.shape[-1]

    def __call__(self, batch: List[dict]) -> dict:
        out = super().__call__(batch)
        ehr_rows = [item['ehr_row'] for item in batch]
        ret_ehr_list = []
        ret_ecg_list = []
        for row in ehr_rows:
            if row >= 0:
                ret_ehr_list.append(self.ret_ehr_embs[row].mean(axis=0))
                ret_ecg_list.append(self.ret_ecg_embs[row].mean(axis=0))
            else:
                ret_ehr_list.append(np.zeros(self._ehr_dim, dtype=np.float32))
                ret_ecg_list.append(np.zeros(self._ecg_dim, dtype=np.float32))
        out['ret_ehr'] = torch.from_numpy(np.stack(ret_ehr_list).astype(np.float32))
        out['ret_ecg'] = torch.from_numpy(np.stack(ret_ecg_list).astype(np.float32))
        return out

def prepare_inputs_5tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, tok_map: Dict[int, torch.Tensor], embed_fn) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    special_ids = set(tok_map.keys())
    B = input_ids.size(0)
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

def _get_base_model(model) -> PrismMedGemma:
    from torch.nn.parallel import DistributedDataParallel as DDP
    if hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model.module if isinstance(model, DDP) else model

def save_checkpoint(base_model: PrismMedGemma, ret_ehr_projector: nn.Module, ret_ecg_projector: nn.Module, cls_head: nn.Linear, tokenizer, output_dir: str, step: int, feature_cols: list, cat_cols: list, cat_cards: list, repr_dim: int, extra_tag: str='') -> str:
    tag = f'_step{step}' if step else extra_tag or '_final'
    ckpt_dir = os.path.join(output_dir, f'checkpoint{tag}')
    os.makedirs(ckpt_dir, exist_ok=True)
    base_model.language_model.save_pretrained(ckpt_dir)
    pass
    try:
        base_model.language_model.config.save_pretrained(ckpt_dir)
    except Exception:
        pass
    try:
        if hasattr(base_model.language_model, 'generation_config'):
            base_model.language_model.generation_config.save_pretrained(ckpt_dir)
    except Exception:
        pass
    tokenizer.save_pretrained(ckpt_dir)
    weights: dict = {}
    for name, p in base_model.named_parameters():
        if name.startswith('language_model.'):
            continue
        weights[name] = p.detach().cpu()
    weights['ret_ehr_projector'] = {k: v.detach().cpu() for k, v in ret_ehr_projector.state_dict().items()}
    weights['ret_ecg_projector'] = {k: v.detach().cpu() for k, v in ret_ecg_projector.state_dict().items()}
    weights['cls_head'] = {k: v.detach().cpu() for k, v in cls_head.state_dict().items()}
    weights['feature_cols'] = feature_cols
    weights['cat_feature_cols'] = cat_cols
    weights['cat_cardinalities'] = cat_cards
    weights['repr_dim'] = repr_dim
    weights['step'] = step
    torch.save(weights, os.path.join(ckpt_dir, 'stage1_weights.pt'))
    pass
    return ckpt_dir

def run_epoch(model, ret_ehr_projector: nn.Module, ret_ecg_projector: nn.Module, loader: DataLoader, optimiser, scaler, cfg: Stage2RetConfig, trainable: list, total_steps: int, global_step: int, ret_ehr_token_id: int, ret_ecg_token_id: int, train: bool=True, cls_head: Optional[nn.Module]=None, tokenizer=None, ckpt_fn=None, skip_batches: int=0) -> Tuple[float, int]:
    base = _get_base_model(model)
    if train:
        model.train()
        base.ecg_tower.eval()
    else:
        model.eval()
        ret_ehr_projector.eval()
        ret_ecg_projector.eval()
    total_loss = 0.0
    n_batches = 0
    is_main = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    ctx = torch.cuda.amp.autocast(enabled=cfg.bf16 and cfg.device.startswith('cuda'), dtype=torch.bfloat16)
    with torch.set_grad_enabled(train):
        _iter = tqdm(loader, leave=False) if is_main and train else loader
        for step, batch in enumerate(_iter):
            if step < skip_batches:
                continue
            ids = batch['input_ids'].to(cfg.device)
            attn = batch['attention_mask'].to(cfg.device)
            lbs = batch['labels'].to(cfg.device)
            ehr = batch['ehr_feat'].to(cfg.device)
            cls_labels = batch['cls_labels'].to(cfg.device)
            ret_ehr = batch['ret_ehr'].to(cfg.device)
            ret_ecg = batch['ret_ecg'].to(cfg.device)
            with torch.no_grad():
                if batch['ecg_embeddings'] is not None:
                    ecg_emb_cur = batch['ecg_embeddings'].to(cfg.device)
                    use_cached_ecg = True
                else:
                    ecg_waves = batch['ecg_waveforms'].to(cfg.device)
                    use_cached_ecg = False
                ecg_emb_list = []
                days_l = []
                for pre_emb, pw, days in zip(batch['prior_ecg_embs_list'], batch['prior_ecg_waves_list'], batch['days_list']):
                    if pre_emb is not None:
                        ecg_emb_list.append(pre_emb.to(cfg.device))
                    elif pw is not None:
                        ecg_emb_list.append(base.ecg_tower(pw.to(cfg.device)))
                    else:
                        ecg_emb_list.append(None)
                    days_l.append(days.to(cfg.device) if days is not None else None)
            with ctx:
                with torch.no_grad():
                    if use_cached_ecg:
                        ecg_soft = base.ecg_projector(ecg_emb_cur.to(base.ecg_projector.net[0].weight.dtype))
                    else:
                        ecg_feat = base.ecg_tower(ecg_waves)
                        ecg_soft = base.ecg_projector(ecg_feat.to(base.ecg_projector.net[0].weight.dtype))
                    ehr_soft = base.encode_ehr(ehr)
                    hist_soft = base.encode_ecg_history(ecg_emb_list, days_l)
                proj_dtype = ret_ehr_projector.net[0].weight.dtype
                ret_ehr_soft = ret_ehr_projector(ret_ehr.to(proj_dtype))
                ret_ecg_soft = ret_ecg_projector(ret_ecg.to(proj_dtype))
                tok_map = {base.ecg_token_id: ecg_soft, base.ehr_token_id: ehr_soft, base.hist_token_id: hist_soft, ret_ehr_token_id: ret_ehr_soft, ret_ecg_token_id: ret_ecg_soft}
                embed_fn = base.get_embed_tokens()
                new_embeds, new_labels, new_attn = prepare_inputs_5tokens(ids, attn, lbs, tok_map, embed_fn)
                out = base.language_model(inputs_embeds=new_embeds, attention_mask=new_attn, labels=new_labels, output_hidden_states=cls_head is not None)
                loss = out.loss
                cls_loss_val = 0.0
                cls_acc_val = float('nan')
                if cls_head is not None:
                    hidden = out.hidden_states[-1]
                    B_ = hidden.size(0)
                    S_ = hidden.size(1)
                    prompt_lens = []
                    for b in range(B_):
                        non_ign = (new_labels[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                        if len(non_ign) > 0:
                            prompt_lens.append(non_ign[0].item())
                        else:
                            prompt_lens.append(S_)
                    pool_mask = torch.zeros(B_, S_, 1, device=hidden.device, dtype=hidden.dtype)
                    for b in range(B_):
                        pool_mask[b, :prompt_lens[b], 0] = 1.0
                    pool_mask = pool_mask * new_attn.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * pool_mask).sum(1) / pool_mask.sum(1).clamp(min=1)
                    logits = cls_head(pooled.to(cls_head.weight.dtype)).squeeze(-1)
                    if is_main and step == 0 and (tokenizer is not None):
                        seq_len = new_labels.shape[1]
                        pl0 = prompt_lens[0]
                        n_pooled = int(pool_mask[0].sum().item())
                        ans_pos = (new_labels[0] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                        gt_txt = tokenizer.decode(new_labels[0, ans_pos].cpu(), skip_special_tokens=True).strip() if ans_pos.numel() > 0 else '(none)'
                        p_vec = pooled[0].float()
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                        pass
                    valid = cls_labels != -1
                    if valid.any():
                        cls_loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], cls_labels[valid].float())
                        loss = loss + cfg.cls_alpha * cls_loss
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
                    if is_main and global_step % cfg.log_interval == 0:
                        avg = total_loss / max(n_batches, 1)
                        cls_info = ''
                        if cls_head is not None:
                            n_yes = (cls_labels == 1).sum().item()
                            n_no = (cls_labels == 0).sum().item()
                            n_unk = (cls_labels == -1).sum().item()
                            cls_info = f'  cls_loss={cls_loss_val:.4f}  cls_acc={cls_acc_val:.3f}  labels=[{n_yes}×Yes/{n_no}×No/{n_unk}×unk]'
                        pass
                        if tokenizer is not None:
                            lbl0 = new_labels[0]
                            ans_pos = (lbl0 != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                            if ans_pos.numel() > 0:
                                gt_ids = lbl0[ans_pos]
                                pred_ids = out.logits[0, ans_pos - 1, :].argmax(-1)
                                gt_txt = tokenizer.decode(gt_ids, skip_special_tokens=False)
                                pred_txt = tokenizer.decode(pred_ids, skip_special_tokens=False)
                                pass
                                pass
                    if is_main and cfg.ckpt_every > 0 and (global_step % cfg.ckpt_every == 0) and (ckpt_fn is not None):
                        ckpt_fn(global_step)
            total_loss += loss.item()
            n_batches += 1
    return (total_loss / max(n_batches, 1), global_step)

def train(cfg: Stage2RetConfig) -> None:
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
        pass
        pass
        pass
        pass
        pass
        pass
        pass
    trainval_mat, _, feature_cols, cat_cols, cat_cards = build_ehr_matrix(cfg.train_val_csv, cfg.test_csv, cfg.patient_json)
    prior_ecg_idx = _extract_prior_ecg_index(cfg.patient_json)
    if is_main:
        pass
    ret_ehr_path = os.path.join(cfg.faiss_dir, 'retrieved_ehr_top3.npy')
    ret_ecg_path = os.path.join(cfg.faiss_dir, 'retrieved_ecg_top3.npy')
    if not os.path.isfile(ret_ehr_path):
        raise FileNotFoundError(f'retrieved_ehr_top3.npy not found in {cfg.faiss_dir}')
    if not os.path.isfile(ret_ecg_path):
        raise FileNotFoundError(f'retrieved_ecg_top3.npy not found in {cfg.faiss_dir}')
    ret_ehr_embs = np.load(ret_ehr_path)
    ret_ecg_embs = np.load(ret_ecg_path)
    if is_main:
        pass
    ret_ehr_dim = ret_ehr_embs.shape[-1]
    ret_ecg_dim = ret_ecg_embs.shape[-1]
    train_idx = LazyJSONIndex(cfg.train_json)
    val_idx = LazyJSONIndex(cfg.val_json)
    if is_main:
        pass
    tokenizer = AutoTokenizer.from_pretrained(cfg.medgemma)
    tokenizer.add_tokens([DEFAULT_ECG_TOKEN, DEFAULT_EHR_TOKEN, DEFAULT_ECG_HIST_TOKEN, DEFAULT_RET_EHR_TOKEN, DEFAULT_RET_ECG_TOKEN], special_tokens=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    ret_ehr_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_RET_EHR_TOKEN)
    ret_ecg_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_RET_ECG_TOKEN)
    if is_main:
        pass
    if is_main:
        pass
    ckpt = torch.load(cfg.stage1_ckpt, map_location='cpu', weights_only=False)
    s1_feature_cols = ckpt.get('feature_cols', feature_cols)
    s1_cat_cols = ckpt.get('cat_feature_cols', cat_cols)
    s1_cat_cards = ckpt.get('cat_cardinalities', cat_cards)
    s1_repr_dim = ckpt.get('repr_dim', cfg.repr_dim)
    s1_step = ckpt.get('global_step', 0)
    if is_main:
        pass
    model_args = PrismModelArgs(medgemma=cfg.medgemma, feature_cols=s1_feature_cols, cat_feature_cols=s1_cat_cols, cat_cardinalities=s1_cat_cards, repr_dim=s1_repr_dim, ehr_d_model=cfg.ehr_d_model, ehr_n_heads=cfg.ehr_n_heads, ehr_n_layers=cfg.ehr_n_layers, n_ecg_tokens=cfg.n_ecg_tokens, n_ehr_tokens=cfg.n_ehr_tokens, n_hist_tokens=cfg.n_hist_tokens, proj_hidden=cfg.proj_hidden)
    model = PrismMedGemma(model_args, tokenizer, bf16=cfg.bf16).to(cfg.device)
    PrismMedGemma.load_stage1_checkpoint(model, cfg.stage1_ckpt)
    freeze_stage1_components(model)
    if cfg.resume_from:
        if is_main:
            pass
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(model.language_model, cfg.resume_from, is_trainable=True)
    else:
        apply_lora_to_language_model(model, cfg)
    d_llm = model.language_model.config.hidden_size
    ret_ehr_projector = build_soft_token_projector(ret_ehr_dim, d_llm, cfg.n_ret_ehr_tokens, cfg.proj_hidden).to(cfg.device)
    ret_ecg_projector = build_soft_token_projector(ret_ecg_dim, d_llm, cfg.n_ret_ecg_tokens, cfg.proj_hidden).to(cfg.device)
    if cfg.bf16:
        ret_ehr_projector = ret_ehr_projector.to(torch.bfloat16)
        ret_ecg_projector = ret_ecg_projector.to(torch.bfloat16)
    if is_main:
        n_p = sum((p.numel() for p in ret_ehr_projector.parameters()))
        pass
        n_p = sum((p.numel() for p in ret_ecg_projector.parameters()))
        pass
    cls_head = nn.Linear(d_llm, 1, bias=True).to(cfg.device)
    if 'cls_head' in ckpt:
        cls_head.load_state_dict(ckpt['cls_head'])
        if is_main:
            pass
    else:
        nn.init.zeros_(cls_head.weight)
        nn.init.zeros_(cls_head.bias)
    cls_head.requires_grad_(True)
    _resume_step = 0
    if cfg.resume_from:
        resume_s1 = os.path.join(cfg.resume_from, 'stage1_weights.pt')
        if is_main:
            pass
        res = torch.load(resume_s1, map_location='cpu', weights_only=False)
        ret_ehr_projector.load_state_dict(res['ret_ehr_projector'])
        ret_ecg_projector.load_state_dict(res['ret_ecg_projector'])
        cls_head.load_state_dict(res['cls_head'])
        _resume_step = int(0)
        if is_main:
            pass
        del res
    if is_main:
        n_trainable = sum((p.numel() for p in model.parameters() if p.requires_grad)) + sum((p.numel() for p in ret_ehr_projector.parameters())) + sum((p.numel() for p in ret_ecg_projector.parameters())) + sum((p.numel() for p in cls_head.parameters()))
        n_total = sum((p.numel() for p in model.parameters()))
        pass
        pass
        pass
    ecg_cache = None
    if cfg.ecg_cache and os.path.isdir(cfg.ecg_cache):
        ecg_cache = ECGCache(cfg.ecg_cache)
    collate_fn = RetCollateClass(h5_root=cfg.h5_root, tokenizer=tokenizer, max_seq_len=cfg.max_seq_len, ret_ehr_embs=ret_ehr_embs, ret_ecg_embs=ret_ecg_embs, ecg_cache=ecg_cache)
    ds_train = RetDataset(train_idx, trainval_mat, prior_ecg_idx)
    ds_val = RetDataset(val_idx, trainval_mat, prior_ecg_idx)
    if cfg.val_max_samples > 0 and len(ds_val) > cfg.val_max_samples:
        from torch.utils.data import Subset
        rng = np.random.default_rng(cfg.seed)
        val_subset = rng.choice(len(ds_val), cfg.val_max_samples, replace=False).tolist()
        val_subset.sort()
        ds_val = Subset(ds_val, val_subset)
        if is_main:
            pass
    if cfg.max_samples > 0 and len(ds_train) > cfg.max_samples:
        from torch.utils.data import Subset
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
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable += list(ret_ehr_projector.parameters())
    trainable += list(ret_ecg_projector.parameters())
    trainable += list(cls_head.parameters())
    total_steps = len(dl_train) * cfg.epochs // cfg.grad_accum
    optimiser = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    optimiser.zero_grad()
    if is_main:
        pass
        pass
    base_model = _get_base_model(model)
    _prev_ckpt: list = []

    def _save_ckpt(step: int, tag: str='') -> str:
        d = save_checkpoint(base_model, ret_ehr_projector, ret_ecg_projector, cls_head, tokenizer, cfg.output_dir, step, s1_feature_cols, s1_cat_cols, s1_cat_cards, s1_repr_dim, extra_tag=tag)
        return d

    def _periodic_save(step: int) -> None:
        d = _save_ckpt(step)
        if _prev_ckpt and os.path.isdir(_prev_ckpt[0]):
            import shutil
            shutil.rmtree(_prev_ckpt[0])
            pass
        _prev_ckpt[:] = [d]
    best_val = float('inf')
    global_step = _resume_step
    n_train_batches = len(dl_train)
    _skip_first_epoch = 0
    if cfg.resume_from and global_step > 0:
        _processed_batches = global_step * cfg.grad_accum
        _skip_first_epoch = _processed_batches % n_train_batches
        if is_main and _skip_first_epoch > 0:
            pass
    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            pass
        skip = _skip_first_epoch if epoch == 1 else 0
        tr_loss, global_step = run_epoch(model, ret_ehr_projector, ret_ecg_projector, dl_train, optimiser, scaler, cfg, trainable, total_steps, global_step, ret_ehr_token_id, ret_ecg_token_id, train=True, cls_head=cls_head, tokenizer=tokenizer, ckpt_fn=_periodic_save if is_main else None, skip_batches=skip)
        val_loss, _ = run_epoch(model, ret_ehr_projector, ret_ecg_projector, dl_val, None, None, cfg, trainable, total_steps, 0, ret_ehr_token_id, ret_ecg_token_id, train=False, cls_head=cls_head)
        if is_main:
            pass
            if val_loss < best_val:
                best_val = val_loss
                best_dir = _save_ckpt(0, tag='_best')
                pass
    if is_main:
        final_dir = _save_ckpt(global_step)
        pass
        pass
    if world_size > 1:
        dist.destroy_process_group()

def parse_args() -> Stage2RetConfig:
    p = argparse.ArgumentParser(description='PRISM Stage 2 + FAISS Retrieval Augmentation')
    p.add_argument('--stage1_ckpt', required=True)
    p.add_argument('--resume_from', default='', help='Resume from a Stage 2 retrieval checkpoint directory (adapter_model.safetensors + stage1_weights.pt). Restores LoRA, ret_projectors, cls_head, and global_step.')
    p.add_argument('--faiss_dir', required=True, help='Directory with retrieved_ehr_top3.npy and retrieved_ecg_top3.npy')
    p.add_argument('--train_json', required=True)
    p.add_argument('--val_json', required=True)
    p.add_argument('--train_val_csv', required=True)
    p.add_argument('--test_csv', required=True)
    p.add_argument('--patient_json', required=True)
    p.add_argument('--h5_root', required=True)
    p.add_argument('--ecg_cache', default='')
    p.add_argument('--medgemma', default='google/medgemma-4b-it')
    p.add_argument('--output_dir', default='./stage2_ret_out')
    p.add_argument('--repr_dim', type=int, default=512)
    p.add_argument('--ehr_d_model', type=int, default=192)
    p.add_argument('--ehr_n_heads', type=int, default=4)
    p.add_argument('--ehr_n_layers', type=int, default=3)
    p.add_argument('--n_ecg_tokens', type=int, default=8)
    p.add_argument('--n_ehr_tokens', type=int, default=8)
    p.add_argument('--n_hist_tokens', type=int, default=8)
    p.add_argument('--proj_hidden', type=int, default=1024)
    p.add_argument('--n_ret_ehr_tokens', type=int, default=8, help='Soft tokens per retrieved EHR context')
    p.add_argument('--n_ret_ecg_tokens', type=int, default=8, help='Soft tokens per retrieved ECG context')
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--grad_accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-05)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_steps', type=int, default=100)
    p.add_argument('--max_seq_len', type=int, default=2048)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cls_alpha', type=float, default=0.1)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--ckpt_every', type=int, default=5000)
    p.add_argument('--max_samples', type=int, default=0)
    p.add_argument('--val_max_samples', type=int, default=10000, help='Max val samples per epoch (0=full). Random subset, seeded.')
    a = p.parse_args()
    cfg = Stage2RetConfig()
    for k, v in vars(a).items():
        setattr(cfg, k, v)
    return cfg
if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)

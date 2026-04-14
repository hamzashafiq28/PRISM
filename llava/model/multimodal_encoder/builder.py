import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .ecg_tower import ECGTower
from .ehr_tower import EHRTower
from .ecg_history_tower import ECGHistoryTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion') or ('ShareGPT4V' in vision_tower):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_ecg_tower(model_args, delay_load: bool=False) -> ECGTower:
    delay = delay_load or getattr(model_args, 'delay_load', False)
    return ECGTower(ecg_ckpt=model_args.ecg_tower, ecg_cfg=model_args.ecg_cfg, delay_load=delay)

def build_ehr_tower(model_args) -> EHRTower:
    return EHRTower(feature_cols=model_args.feature_cols, cat_feature_cols=model_args.cat_feature_cols, cat_cardinalities=model_args.cat_cardinalities, repr_dim=getattr(model_args, 'repr_dim', 512), d_model=getattr(model_args, 'ehr_d_model', 192), n_heads=getattr(model_args, 'ehr_n_heads', 4), n_layers=getattr(model_args, 'ehr_n_layers', 3), dropout=getattr(model_args, 'ehr_dropout', 0.1))

def build_history_tower(model_args) -> ECGHistoryTower:
    return ECGHistoryTower(ecg_dim=model_args.ecg_dim, repr_dim=getattr(model_args, 'repr_dim', 512), dropout=getattr(model_args, 'ehr_dropout', 0.1))

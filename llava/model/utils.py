from transformers import AutoConfig

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        pass
        pass
        confirm = input('Please confirm that you want to upgrade the checkpoint. [Y/N]')
        if confirm.lower() in ['y', 'yes']:
            pass
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, 'model_type', 'llava')
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            pass
        else:
            pass
            exit(1)

import os
import re
import warnings
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from arguments import model_config, cache_dir


DEFAULT_SD_DOWNLOAD_URL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt"
DEFAULT_SDXL_DOWNLOAD_URL = 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors'


def dummy_checker(images, **kwargs): 
	return images, [False for _ in images]


class SDModel(object):

    def __init__(self):
        self.lora_mode = model_config.lora_mode
        self.pipeline, self.lora_usage = self.load_model()
        self.safety_checker = self.pipeline.safety_checker

    @staticmethod
    def load_model():
        if model_config.type == 'SDXL':
            PipelineClass = StableDiffusionXLPipeline
        else:
            PipelineClass = StableDiffusionPipeline

        sd_cache_dir = os.path.join(cache_dir, 'stablediffusion')
        if not os.path.exists(sd_cache_dir):
            os.makedirs(sd_cache_dir)

        # parsing model config
        kwargs = {}
        if model_config.fp16:
            kwargs['torch_dtype'] = torch.float16
        kwargs['cache_dir'] = sd_cache_dir

        def check_path(path):
            return os.path.exists(path) and os.path.isfile(path)
        
        if model_config.model_name is None:
            if model_config.type == 'SDXL':
                model_config.model_name = 'Default SDXL'
                single_file_path = DEFAULT_SDXL_DOWNLOAD_URL
            else:
                model_config.model_name = 'Default SD'
                single_file_path = DEFAULT_SD_DOWNLOAD_URL
        else:
            single_file_path = None
            model_dir = os.path.join(os.getcwd(), 'models', 'Stable-diffusion')
            for temp_path in [
                model_config.model_name,
                os.path.join(model_dir, f'{model_config.model_name}.safetensors'),
                os.path.join(model_dir, f'{model_config.model_name}.ckpt')
            ]:
                if check_path(temp_path):
                    single_file_path = temp_path


        print(f'Loading base model {model_config.model_name}')
        if single_file_path is not None:
            # kwargs['load_safety_checker'] = model_config.safety_checker
            pipeline = PipelineClass.from_single_file(single_file_path, **kwargs)

        else:
            # if not model_config.safety_checker:
            #     kwargs['safety_checker'] = None
            #     kwargs['requires_safety_checker'] = False
            pipeline = PipelineClass.from_pretrained(model_config.model_name, **kwargs)

        pipeline.to('cuda')
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.enable_attention_slicing()
        
        # load LoRAs
        lora_usage = {}
        pipeline.unload_lora_weights()
        model_config.lora_usage = model_config.lora_usage if model_config.lora_usage is not None else []
        for lora_info in model_config.lora_usage:
            if isinstance(lora_info, dict):
                lora_name, lora_scale = list(lora_info.items())[0]
            else:
                lora_name, lora_scale = lora_info, 1.0

            lora_path = os.path.join(os.getcwd(), 'models', 'LoRA', f'{lora_name}.safetensors')
            if not os.path.exists(lora_path): continue
            
            try:
                print(f'Loading LoRA {lora_name}...')
                pipeline.load_lora_weights(lora_path, adapter_name=lora_name)
                lora_usage[lora_name] = lora_scale
                print(f'Done.')
            except Exception as e:
                warnings.warn(f'Unmatched LoRA {lora_name}')
        
        if model_config.lora_mode == 'fixed':
            pipeline.set_adapters(list(lora_usage.keys()), adapter_weights=list(lora_usage.values()))
            pipeline.fuse_lora(adapter_names=list(lora_usage.keys()))
        
        return pipeline, lora_usage

    def __call__(self, safety_checker=True, **kwargs):
        if isinstance(self.pipeline, StableDiffusionPipeline):
            self.pipeline.safety_checker = self.safety_checker if safety_checker else dummy_checker

        payload = self.parse_prompt(kwargs['prompt'])
        kwargs['prompt'] = payload['prompt']
        if self.lora_mode == 'dynamic':
            self.pipeline.set_adapters(
                list(payload['lora'].keys()), adapter_weights=list(payload['lora'].values()))
        output = self.pipeline(**kwargs)
        return output
    
    def parse_prompt(self, prompt):
        payload = {}
        
        pattern = r'(<lora:([-,\u4e00-\u9fa5\w\s]+):?([01]?\.?\d+)?>)'
        matches = re.findall(pattern, prompt)

        lora_usage = {}
        for match in matches:
            lora_name = match[1]
            if lora_name not in self.lora_usage: continue
            try:
                lora_scale = float(match[2])
                lora_scale = np.clip(lora_scale, 0.0, 1.0)
            except Exception as e:
                lora_scale = 1.0
            lora_usage[lora_name] = lora_scale

        new_prompt = re.sub(pattern, '', prompt)
        
        payload['prompt'] = new_prompt
        payload['lora'] = lora_usage
        
        return payload


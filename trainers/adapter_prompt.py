import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

# Tokenization function with exception handling
def tokenize_prompts(classnames):
    try:
        return torch.cat([clip.tokenize(c) for c in classnames])
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

# Adapter from the first model
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 384, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(384, 768, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# PromptLearner from the second model
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.n_cls = n_cls
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Initialize context vectors
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # Token prefix and suffix (non-trainable parts)
        classnames = [name.replace("_", " ") for name in classnames]
        self.token_prefix = clip_model.token_embedding(clip.tokenize("X")).type(clip_model.dtype)
        self.token_suffix = clip_model.token_embedding(clip.tokenize(".")).type(clip_model.dtype)

        #self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        # Concatenate prefix, trainable context, and suffix
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = prefix[i : i + 1, :, :]
            class_i = suffix[i : i + 1, :name_len, :]
            suffix_i = suffix[i : i + 1, name_len:, :]
            ctx_i = ctx[i : i + 1, :, :]
            prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
            prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        return prompts




# CustomCLIP integrating both Adapter and PromptLearner
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual  # Image encoder from CLIP
        self.text_encoder = clip_model.encode_text  # Text encoder from CLIP

        # Integrating both trainable parts: Adapter and PromptLearner
        self.adapter = Adapter(1024, 4)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, classnames):
        # Text processing: Prompt Tuning
        prompts = self.prompt_learner()
        tokenized_prompts = tokenize_prompts(classnames)
        if tokenized_prompts is None:
            return None
        text_features = self.text_encoder(prompts)

        # Image processing: Adapter after Image Encoder
        image_features = self.image_encoder(image.type(self.dtype))
        adapted_image_features = self.adapter(image_features)

        # Normalizing features
        image_features = adapted_image_features / adapted_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # CLIP-style logits computation
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

# Trainer class combining both models and integrating training for Adapter and PromptLearner
@TRAINER_REGISTRY.register()
class UnifiedTrainer(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()

        print("Building unified CLIP model")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and text encoder (except trainable parts)")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name and "adapter" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected ({torch.cuda.device_count()} GPUs), using DataParallel")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        if self.cfg.TRAINER.COOP.PREC == "amp":
            with autocast():
                output = self.model(image, self.dm.dataset.classnames)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optim)
            scaler.update()
        else:
            output = self.model(image, self.dm.dataset.classnames)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

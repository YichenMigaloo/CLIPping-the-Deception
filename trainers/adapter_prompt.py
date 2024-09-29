import torch
import torch.nn as nn
from clip import clip

# Adapter from Model 1
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

# PromptLearner from Model 2
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # Initialize context vectors
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Token prefix and suffix (non-trainable parts)
        self.token_prefix = clip_model.token_embedding(torch.zeros(1))  # Example placeholder
        self.token_suffix = clip_model.token_embedding(torch.zeros(1))  # Example placeholder
        
    def forward(self):
        # Concatenate prefix, trainable context, and suffix
        return torch.cat([self.token_prefix, self.ctx, self.token_suffix], dim=1)

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
        tokenized_prompts = torch.cat([clip.tokenize(c) for c in classnames])
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

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple , Union
import numpy as np

# LayerNorm

class LayerNorm(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
# QuickGELU

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
    

# ResidualAttentionBlock

class ResidualAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, n_head: int, attn_mask: Union[torch.Tensor, None] = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model)),
            ])
        )
        
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        return self.attn(x,x,x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x
    
# Transformer

class Transformer(nn.Module):
    
    def __init__(self, width:int, layers: int, heads: int, attn_mask: Union[torch.Tensor, None] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)
    
    
# VisionTransformer

class VisionTransformer(nn.Module):
    
    def __init__(self, input_resolution:int , patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias = False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x) # shape [*, width, GridH, GridW]
        x = x.reshape(x.shape[0], x.shape[1], -1) # shape [*, width, GridH * GridW]
        x = x.permute(0, 2, 1) # shape [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros((x.shape[0], 1, x.shape[2]), device=x.device, dtype=x.dtype), x], dim=1) # shape [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding.to(x.dtype) # shape [*, grid ** 2 + 1, width]
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        
        x = self.ln_post(x[:, 0, :]) # take the class token
        
        if self.proj is not None:
            x = x @ self.proj
            
        return x
    

# CLIPVisionTransformer

class CLIP(nn.Module):
    
    def __init__(
        self, 
        embed_dim: int,
        #vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        #text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        
        super().__init__()
        
        self.context_length = context_length
        
        vision_head = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_head,
            output_dim=embed_dim
        )
        
        self.tranformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.initialize_parameters()
        
        
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        proj_std = (self.tranformer.width ** -0.5) * ((2 * self.tranformer.layers) ** -0.5)
        attn_std = self.tranformer.width ** -0.5
        fc_std = (2 * self.tranformer.width) ** -0.5
        for block in self.tranformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.tranformer.width ** -0.5)
            
    def build_attention_mask(self) -> torch.Tensor:
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1) # Zeros out the lower triangle, leaving the upper triangle as -inf
        return mask
    
    @property
    def dtype(self) -> torch.dtype:
        return self.visual.conv1.weight.dtype
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image.type(self.dtype))
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype) # [Batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.tranformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 
        
        return x
    
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # normalize features
        image_features = image_features / image_features.norm(dim = 1, keepdim=True)
        text_features = text_features / text_features.norm(dim = 1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
    
    
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
                
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k","bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
                    
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                tensor = getattr(l, name)
                if tensor is not None:
                    tensor.data = tensor.data.half()
                    
    model.apply(_convert_weights_to_fp16)
                
                
def build_model(state_dict: dict):
    """Builds a CLIP model from a state_dict"""
    
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict.keys() if k.startswith("tranformer.resblocks")))
                             
    
    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )
    
    for key in ["input_resolution", "contexT_lenght", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    
    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    
    return model.eval()

    
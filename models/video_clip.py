import torch
import torch.nn as nn
from clip.model import CLIP
import clip

import numpy as np

from collections import OrderedDict
from pkg_resources import packaging
from typing import Union, List
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


class VideoCLIP(CLIP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = Transformer(
            width=kwargs['transformer_width'],
            layers=kwargs['transformer_layers'],
            heads=kwargs['transformer_heads'])

    def forward(self, video_frames, text, splits, video_neg_frames=None, neg_splits=None, return_features=False, is_train=False):
        device = video_frames[0].device

        if video_neg_frames is None:
            video_features = []
            for i in range(len(video_frames)):
                bs, c, h, w = video_frames[i].shape
                video_frame_features = self.encode_image(video_frames[i])
                video_frame_features = video_frame_features[:bs]
                video_frame_features = video_frame_features.split(splits[i], dim=0)
                video_feature = torch.cat([feat.mean(dim=0, keepdim=True) for feat in video_frame_features])
                video_features.append(video_feature)
            video_features = torch.cat(video_features, dim=0)

        if video_neg_frames is not None:
            video_features = []
            video_neg_features = []
            for i in range(len(video_frames)):
                bs, c, h, w = video_frames[i].shape
                bs_neg, _, _, _ = video_neg_frames.shape
                # concat positive and negative frames
                _video_frames = torch.cat([video_frames[i], video_neg_frames], dim=0)
                if is_train and _video_frames.shape[0] < 192:
                    _video_frames = torch.cat([
                        _video_frames, torch.zeros(192 - _video_frames.shape[0], c, h, w).to(device)], dim=0)
                # encode frames
                _video_frame_features = self.encode_image(_video_frames)
                # split positive and negative features
                video_frame_features = _video_frame_features[:bs]
                video_neg_frame_features = _video_frame_features[bs:bs+bs_neg]
                # average positive features
                video_frame_features = video_frame_features.split(splits[i], dim=0)
                video_feature = torch.cat([feat.mean(dim=0, keepdim=True) for feat in video_frame_features], dim=0)
                video_features.append(video_feature)
                # average negative features
                _neg_splits = torch.tensor(neg_splits)
                _neg_splits = _neg_splits[_neg_splits > 0].tolist()
                if len(_neg_splits) == 0:
                    video_neg_feature = []
                else:
                    video_neg_frame_features = video_neg_frame_features.split(_neg_splits, dim=0)
                    video_neg_feature = [feat.mean(dim=0, keepdim=True) for feat in video_neg_frame_features]
                if len(video_neg_feature) != video_feature.shape[0]:
                    zeros_tensor = torch.zeros([1, video_feature.shape[1]]).to(device).fill_(1e-4)
                    for j in range(len(neg_splits)):
                        if neg_splits[j] == 0:
                            video_neg_feature.insert(j, zeros_tensor)
                video_neg_feature = torch.cat(video_neg_feature, dim=0)
                video_neg_features.append(video_neg_feature)

            video_features = torch.cat(video_features, dim=0)
            video_neg_features = torch.cat(video_neg_features, dim=0)

        text_features = self.encoding_text(text, device)

        if video_neg_frames is not None:
            video_neg_features_list = []
            for r in range(6):
                ratio = 1 - r * 0.1
                _video_neg_features = video_neg_features * ratio + video_features * (1 - ratio)
                _video_neg_features = _video_neg_features / _video_neg_features.norm(dim=1, keepdim=True)
                video_neg_features_list.append(_video_neg_features)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if return_features:
            if video_neg_frames is not None:
                return video_features, text_features, video_neg_features_list
            return video_features, text_features

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * video_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def encoding_text(self, text, device):
        text_embeddings = tokenize(text, context_length=256, truncate=True).to(device)
        text_features = self.encode_text(text_embeddings)
        return text_features

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        positional_embedding = self._interpolate_pos_embed(self.positional_embedding.type(self.dtype), text.shape[1])
        attention_mask = self._build_attention_mask(text.shape[1]).to(x.device)

        x = x + positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attention_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def _interpolate_pos_embed(self, pos_embed, text_length):
        if text_length > 77:
            device = pos_embed.device
            pos_embed = torch.nn.functional.interpolate(
                pos_embed[:, None, None, :].permute(1, 3, 0, 2),
                size=(text_length, 1), mode='bicubic', align_corners=False)[0, :, :, 0].permute(1, 0)
        else:
            pos_embed = pos_embed[:text_length]
        return pos_embed

    def _build_attention_mask(self, text_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(text_length, text_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask.type(self.dtype)


def build_video_clip_model(name, model_path=None, device="cuda"):
    model_path = _MODELS[name] if model_path is None else model_path

    if 'pretrained' in model_path:
        with open(model_path, 'rb') as opened_file:
            model = torch.jit.load(opened_file, map_location="cpu").eval()
            state_dict = model.state_dict()
            opened_file.close()
    else:
        state_dict = torch.load(model_path, map_location="cpu")['model']

    model = build_model(state_dict).to(device)
    if str(device) == "cpu":
        model.float()
    return model


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = VideoCLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution, vision_layers=vision_layers,
        vision_width=vision_width, vision_patch_size=vision_patch_size,
        context_length=context_length, vocab_size=vocab_size,
        transformer_width=transformer_width, transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    max_length = -1
    for i, tokens in enumerate(all_tokens):
        max_length = max(max_length, len(tokens))
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    result = result[:, :max_length]
    return result


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for resblock in self.resblocks:
            x = resblock(x, attn_mask)
        return x
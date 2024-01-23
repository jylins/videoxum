import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

from .vit_video import LocalAttenModule, create_tt


class VTSum_BLIP_TT(nn.Module):
    def __init__(self,
                 tt_depth=1,
                 loss_type='cross_entropy',
                 med_config='configs/med_config.json',
                 vit='base',
                 prompt='a video of ',
                 max_text_length=128):
        """ VTSum_BLIP model with Temporal Transformer (TT)
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            vit (str): model size of vision transformer
            prompt (str): the prompt for the decoder
        """
        super().__init__()

        self.vit = vit
        assert vit in ['base', 'large']
        if vit == 'base':
            self.vision_width = 768
        elif vit == 'large':
            self.vision_width = 1024

        # create temporal transformer
        self.position_embeddings = nn.Embedding(512, self.vision_width)
        self.tt, tt_width = create_tt(
            self.vit, depth=tt_depth)
        assert tt_width == self.vision_width

        # vsum decoder
        self.vsum_head = nn.Sequential(nn.Linear(self.vision_width, 1))
        nn.init.normal_(self.vsum_head[-1].weight, std=.02)
        self.loss_type = loss_type

        # tsum decoder
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = self.vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)

        self.max_text_length = max_text_length
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

    def forward(self, video_embeddings, video_mask, vsum_labels, tsum_labels):
        device = video_embeddings.device

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
        video_embeddings = video_embeddings + position_embeddings

        # temporal transformer
        video_embeddings = self.tt(video_embeddings, video_mask)
        video_embeddings = video_embeddings + position_embeddings

        # vsum decoder
        num_repeat = vsum_labels.shape[1]
        saliency_scores = self.vsum_head(video_embeddings)
        saliency_scores = saliency_scores[:, :, 0].repeat_interleave(num_repeat, dim=0)

        num_frame = saliency_scores.size(-1)
        saliency_scores = saliency_scores.reshape(-1, num_frame)
        vsum_labels = vsum_labels.reshape(-1, num_frame)

        if self.loss_type == 'cross_entropy':
            loss_vsum = self.cross_entropy_loss(
                saliency_scores, vsum_labels, video_mask.repeat_interleave(num_repeat, dim=0))
        elif self.loss_type == 'l2':
            loss_vsum = self.l2_loss(
                saliency_scores, vsum_labels, video_mask.repeat_interleave(num_repeat, dim=0))
        else:
            raise NotImplementedError

        # text decoder
        text = self.tokenizer(
            tsum_labels, padding='longest',
            truncation=True, max_length=self.max_text_length,
            return_tensors="pt").to(device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=video_embeddings,
            encoder_attention_mask=video_mask,
            labels=decoder_targets,
            return_dict=True)

        loss_tsum = decoder_output.loss

        return loss_tsum, loss_vsum

    def generate(self, video_embeddings, video_mask, sample=False,
                 num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        device = video_embeddings.device
        batch_size = video_embeddings.size(0)

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
        video_embeddings = video_embeddings + position_embeddings

        # temporal transformer
        video_embeddings = self.tt(video_embeddings, video_mask)
        video_embeddings = video_embeddings + position_embeddings

        # vsum decoder
        saliency_scores = self.vsum_head(video_embeddings)

        # tsum decoder
        if not sample:
            video_embeddings = video_embeddings.repeat_interleave(num_beams, dim=0)

        model_kwargs = {"encoder_hidden_states": video_embeddings,
                        "encoder_attention_mask": video_mask.repeat_interleave(num_beams, dim=0)}

        prompt = [self.prompt] * batch_size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        tsum_labels = []
        for output in outputs:
            tsum_label = self.tokenizer.decode(output, skip_special_tokens=True)
            tsum_labels.append(tsum_label[len(self.prompt):])
        return tsum_labels, saliency_scores

    def cross_entropy_loss(self, saliency_scores, vsum_labels, video_mask, reduction='mean'):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            saliency_scores, vsum_labels, weight=video_mask, reduction=reduction)
        return loss

    def l2_loss(self, saliency_scores, vsum_labels, video_mask):
        return torch.sum(((saliency_scores.sigmoid() - vsum_labels) * video_mask) ** 2) / video_mask.sum()

    def _interpolate_pos_embed(self, pos_embed, video_length):
        if video_length > 512:
            pos_embed = torch.nn.functional.interpolate(
                pos_embed[:, None, None, :].permute(1, 3, 0, 2),
                size=(video_length, 1), mode='bicubic', align_corners=False)[0, :, :, 0].permute(1, 0)
        else:
            pos_embed = pos_embed[:video_length]
        return pos_embed


class VTSum_BLIP_TT_CA(VTSum_BLIP_TT):
    def __init__(self,
                 kernel_size=5,
                 *args, **kwargs):
        """ VTSum_BLIP model with Temporal Transformer (TT) and Context Aggregation (CA)
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            vit (str): model size of vision transformer
            prompt (str): the prompt for the decoder
        """
        super(VTSum_BLIP_TT_CA, self).__init__(*args, **kwargs)
        self.kernel_size = kernel_size
        self.local_atten_head = LocalAttenModule(
            embed_dim=self.vision_width, kernel_size=kernel_size)

    def forward(self, video_embeddings, video_mask, vsum_labels, tsum_labels):
        device = video_embeddings.device

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
        video_embeddings = video_embeddings + position_embeddings

        # temporal transformer
        video_embeddings = self.tt(video_embeddings, video_mask)
        video_embeddings = video_embeddings + position_embeddings

        # local attention
        video_len = video_embeddings.size(1)
        attention_mask = torch.zeros([video_len, video_len])
        half_atten_len = min(self.kernel_size // 2 + 1, video_len)
        for j in range(half_atten_len):
            attention_mask += torch.diag(torch.ones(video_len - j), diagonal=j)
            if j > 0:
                attention_mask += torch.diag(torch.ones(video_len - j), diagonal=-j)
        attention_mask = attention_mask.to(device)
        attention_mask = attention_mask[None, :, :] * video_mask[:, None, :] * video_mask[:, :, None]
        video_embeddings = self.local_atten_head(video_embeddings, attention_mask[:, None, :, :])

        # vsum decoder
        num_repeat = vsum_labels.shape[1]
        saliency_scores = self.vsum_head(video_embeddings)
        saliency_scores = saliency_scores[:, :, 0].repeat_interleave(num_repeat, dim=0)
        num_frame = saliency_scores.size(-1)
        saliency_scores = saliency_scores.reshape(-1, num_frame)
        vsum_labels = vsum_labels.reshape(-1, num_frame)

        if self.loss_type == 'cross_entropy':
            loss_vsum = self.cross_entropy_loss(
                saliency_scores, vsum_labels, video_mask.repeat_interleave(num_repeat, dim=0))
        elif self.loss_type == 'l2':
            loss_vsum = self.l2_loss(
                saliency_scores, vsum_labels, video_mask.repeat_interleave(num_repeat, dim=0))
        else:
            raise NotImplementedError

        # text decoder
        text = self.tokenizer(
            tsum_labels, padding='longest',
            truncation=True, max_length=self.max_text_length,
            return_tensors="pt").to(device)

        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100

        decoder_output = self.text_decoder(
            input_ids=text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=video_embeddings,
            encoder_attention_mask=video_mask,
            labels=decoder_targets,
            return_dict=True)

        loss_tsum = decoder_output.loss

        return loss_tsum, loss_vsum

    def generate(self, video_embeddings, video_mask, sample=False,
                 num_beams=3, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
        device = video_embeddings.device
        batch_size = video_embeddings.size(0)

        # interpolate position embeddings
        position_embeddings = self._interpolate_pos_embed(self.position_embeddings.weight, video_embeddings.size(1))
        video_embeddings = video_embeddings + position_embeddings

        # temporal transformer
        video_embeddings = self.tt(video_embeddings, video_mask)
        video_embeddings = video_embeddings + position_embeddings

        # local attention
        video_len = video_embeddings.size(1)
        attention_mask = torch.zeros([video_len, video_len])
        half_atten_len = min(self.kernel_size // 2 + 1, video_len)
        for j in range(half_atten_len):
            attention_mask += torch.diag(torch.ones(video_len - j), diagonal=j)
            if j > 0:
                attention_mask += torch.diag(torch.ones(video_len - j), diagonal=-j)
        attention_mask = attention_mask.to(device)
        attention_mask = attention_mask[None, :, :] * video_mask[:, None, :] * video_mask[:, :, None]
        video_embeddings = self.local_atten_head(video_embeddings, attention_mask[:, None, :, :])

        # vsum decoder
        saliency_scores = self.vsum_head(video_embeddings)

        # tsum decoder
        if not sample:
            video_embeddings = video_embeddings.repeat_interleave(num_beams, dim=0)

        model_kwargs = {"encoder_hidden_states": video_embeddings,
                        "encoder_attention_mask": video_mask.repeat_interleave(num_beams, dim=0)}

        prompt = [self.prompt] * batch_size
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        tsum_labels = []
        for output in outputs:
            tsum_label = self.tokenizer.decode(output, skip_special_tokens=True)
            tsum_labels.append(tsum_label[len(self.prompt):])
        return tsum_labels, saliency_scores


def v2vt_sum(version='vtsum_blip_tt', pretrained='', **kwargs):
    assert version in ['vtsum_blip_tt', 'vtsum_blip_tt_ca']
    if version == 'vtsum_blip_tt':
        model = VTSum_BLIP_TT(**kwargs)
    elif version == 'vtsum_blip_tt_ca':
        model = VTSum_BLIP_TT_CA(**kwargs)
    if pretrained:
        if 'pretrained' in pretrained:
            model, msg = load_blip_pretrained_checkpoint(model, pretrained)
            print(f'missing keys: {msg.missing_keys}')
        else:
            model, msg = load_checkpoint(model, pretrained)
            print(f'missing keys: {msg.missing_keys}')
    return model


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'load checkpoint from {url_or_filename}')
    return model, msg


def load_blip_pretrained_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint['model']
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape != model.state_dict()[key].shape:
                del state_dict[key]
        if key == 'position_embeddings.weight':
            state_dict[key] = state_dict['text_decoder.bert.embeddings.position_embeddings.weight']

    msg = model.load_state_dict(state_dict, strict=False)
    print(f'load checkpoint from {url_or_filename}')
    return model, msg

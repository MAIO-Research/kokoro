import torch
from torch import nn
from munch import Munch

from models import TextEncoder


class PlainTextBERT(nn.Module):
    """
    Drop-in replacement for PL-BERT for languages without a pretrained
    PL-BERT checkpoint (e.g. Armenian). Reuses the same CNN+LSTM TextEncoder
    architecture as the model's main text encoder, trained from scratch
    jointly with the rest of the model rather than pretrained separately on
    raw text. Consumes the same phoneme ids as `model.text_encoder` -- no
    separate tokenizer/vocab is needed.
    """

    def __init__(self, n_token, hidden_size=512, n_layer=3, max_position_embeddings=512):
        super().__init__()
        self.encoder = TextEncoder(channels=hidden_size, kernel_size=5, depth=n_layer, n_symbols=n_token)
        self.config = Munch(hidden_size=hidden_size, max_position_embeddings=max_position_embeddings)

    def forward(self, input_ids, attention_mask):
        input_lengths = attention_mask.sum(dim=1).long()
        text_mask = ~attention_mask.bool()
        out = self.encoder(input_ids, input_lengths, text_mask)  # [B, C, T]
        return out.transpose(1, 2)  # [B, T, C], matching PL-BERT's output convention


def load_plbert(model_params):
    return PlainTextBERT(
        n_token=model_params.n_token,
        hidden_size=model_params.hidden_dim,
        n_layer=model_params.n_layer,
    )

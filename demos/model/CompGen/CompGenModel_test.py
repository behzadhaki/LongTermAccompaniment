# =================================================================================================
# Load Mega dataset as torch.utils.data.Dataset
'''from data import Groove2Drum2BarDataset

# load dataset as torch.utils.data.Dataset
training_dataset = Groove2Drum2BarDataset(
    dataset_bz2_filepath="data/dataset_json_settings/4_4_Beats_MegaSet.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    should_upsample_to_ensure_genre_balance=True,
    collapse_tapped_sequence=True,
    use_cached=True,
    augment_dataset=True,
    move_all_to_gpu=False
)

'''

import torch

# ==============================================================================
# Input Groove Layer
# ===============================================================================
from model.CompGenVAE.components import InputGrooveLayer


igl = InputGrooveLayer(
    embedding_size=3,
    d_model=128,
    max_len=32,
    velocity_dropout=0.1,
    offset_dropout=0.1,
    positional_encoding_dropout=0.1
)

input_batch = torch.rand((64, 32, 3))
tempo = torch.ones((64)) * 0.24
t_ = igl.forward(input_batch)

# ==============================================================================
# Encoder
# ===============================================================================
from model.CompGenVAE.components import Encoder

encoder = Encoder(
    d_model=128,
    nhead=4,
    dim_feedforward=512,
    num_encoder_layers=3,
    dropout=0.1
)

t__ = encoder.forward(t_)
t__.shape

# ==============================================================================
# Reparameterization
# ===============================================================================
from model.CompGenVAE.components import LatentLayer

latent_layer = LatentLayer(
    max_len=32+1,           # +1 for the tempo appended to the end of input sequence
    d_model=128,
    latent_dim=16
)

mu, log_var, z = latent_layer.forward(t__)

mu.shape, log_var.shape, z.shape

# ==============================================================================
# GenreLayer
# ===============================================================================

from model.CompGenVAE.components import GenreLayer

genre_layer = GenreLayer(
    max_len=32+1,           # +1 for the tempo appended to the end of input sequence
    d_model=128,
    n_genres=12
)

genre_logits = genre_layer.forward(t__)

genre_logits.shape

# ==============================================================================
# ComplexityLayer
# ===============================================================================
from model.CompGenVAE.components import ComplexityLayer

complexity_layer = ComplexityLayer(
    max_len=32+1,           # +1 for the tempo appended to the end of input sequence
    d_model=128
)

complexity_logit = complexity_layer.forward(t__)
complexity_logit.shape

# ==============================================================================
# CompGenDecoderInput
# ===============================================================================
from model.CompGenVAE.components import CompGenDecoderInput

decoder_input = CompGenDecoderInput(
    max_len=32,
    latent_dim=16,
    d_model=128,
    n_genres=12,
)

tempos_normalized = torch.ones((64)) * 0.24
dec_in_ = decoder_input.forward(
    latent_z=z,
    tempo_normalized=tempos_normalized,
    genre_logits=genre_logits,
    complexity_logits=complexity_logit
)


# ==============================================================================
# CompGenDecoder
# ===============================================================================
decoder = Encoder(
    d_model=128,
    nhead=4,
    dim_feedforward=512,
    num_encoder_layers=3,
    dropout=0.1
)


dec_out_ = decoder.forward(dec_in_)

# ==============================================================================
# OutputLayer
# ===============================================================================
from model.CompGenVAE.components import OutputLayer

output_layer = OutputLayer(
    embedding_size=27,
    d_model=128
)

h_logits, v_logits, o_logits, hvo_logits = output_layer.forward(dec_out_)
h_logits.shape, v_logits.shape, o_logits.shape, hvo_logits.shape

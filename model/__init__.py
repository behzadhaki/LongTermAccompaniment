#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu

import torch
import math


# --------------------------------------------------------------------------------
# ------------       Positinal Encoding BLOCK                ---------------------
# --------------------------------------------------------------------------------
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class InputGrooveRhythmLayer(torch.nn.Module):
    """
    Receives a hvo tensor of shape (batch, max_len, 3), and returns a tensor of shape
    (batch, 33, d_model)

    """

    def __init__(self, embedding_size, d_model, max_len,
                 velocity_dropout, offset_dropout,
                 positional_encoding_dropout, has_velocity=True, has_offset=True):
        super(InputGrooveRhythmLayer, self).__init__()
        self.velocity_dropout = torch.nn.Dropout(p=velocity_dropout)
        self.offset_dropout = torch.nn.Dropout(p=offset_dropout)
        self.HitsLinear = torch.nn.Linear(embedding_size//3, d_model, bias=True)
        self.VelocitiesLinear = torch.nn.Linear(embedding_size//3, d_model, bias=True)
        self.OffsetsLinear = torch.nn.Linear(embedding_size//3, d_model, bias=True)
        self.HitsReLU = torch.nn.ReLU()
        self.VelocitiesReLU = torch.nn.ReLU()
        self.OffsetsReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, (max_len), positional_encoding_dropout)
        self.has_velocity = has_velocity
        self.has_offset = has_offset

    def init_weights(self, initrange=0.1):
        self.HitsLinear.bias.data.zero_()
        self.HitsLinear.weight.data.uniform_(-initrange, initrange)
        self.VelocitiesLinear.bias.data.zero_()
        self.VelocitiesLinear.weight.data.uniform_(-initrange, initrange)
        self.OffsetsLinear.bias.data.zero_()
        self.OffsetsLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, hvo, ):
        '''

        :param hvo: shape (batch, 32, 3)
        :return:
        '''

        if self.has_velocity and self.has_offset:
            n_dim = hvo.shape[2] // 3
            hit = hvo[:, :, :n_dim]
            vel = hvo[:, :, n_dim:2*n_dim]
            offset = hvo[:, :, 2*n_dim:]
            # hvo_ = torch.cat((hit, self.velocity_dropout(vel), self.offset_dropout(offset)), dim=2)
            hits_projection = self.HitsReLU(self.HitsLinear(hit))
            velocities_projection = self.VelocitiesReLU(self.VelocitiesLinear(self.velocity_dropout(vel)))
            offsets_projection = self.OffsetsReLU(self.OffsetsLinear(self.offset_dropout(offset)))
            hvo_projection = hits_projection + velocities_projection + offsets_projection
            out = self.PositionalEncoding(hvo_projection)
        elif self.has_velocity:
            n_dim = hvo.shape[2] // 2
            hit = hvo[:, :, :n_dim]
            vel = hvo[:, :, n_dim:]
            hits_projection = self.HitsReLU(self.HitsLinear(hit))
            velocities_projection = self.VelocitiesReLU(self.VelocitiesLinear(self.velocity_dropout(vel)))
            hvo_projection = hits_projection + velocities_projection
            out = self.PositionalEncoding(hvo_projection)
        elif self.has_offset:
            n_dim = hvo.shape[2] // 2
            hit = hvo[:, :, :n_dim]
            offset = hvo[:, :, n_dim:]
            hits_projection = self.HitsReLU(self.HitsLinear(hit))
            offsets_projection = self.OffsetsReLU(self.OffsetsLinear(self.offset_dropout(offset)))
            hvo_projection = hits_projection + offsets_projection
            out = self.PositionalEncoding(hvo_projection)
        else:
            hit = hvo
            hits_projection = self.HitsReLU(self.HitsLinear(hit))
            hvo_projection = hits_projection
            out = self.PositionalEncoding(hvo_projection)

        return out, hit[:, :, 0], hvo_projection


# --------------------------------------------------------------------------------
# ------------                  ENCODER BLOCK                ---------------------
# --------------------------------------------------------------------------------

class TransformerEncoder(torch.nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, max_len, auto_regressive=False):
        """Transformer Encoder Layers Wrapped into a Single Module"""
        super(TransformerEncoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        self.auto_regressive = auto_regressive

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)

        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=norm_encoder)

        self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(max_len)

    # def generate_square_subsequent_mask(self, sz, device):
    #     mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def forward(self, src, src_key_padding_mask):
        """
        input and output both have shape (batch, seq_len, embed_dim)
        :param src:
        :return:
        """
        if self.auto_regressive:
            if self.causal_mask.device != src.device:
                self.causal_mask = self.causal_mask.to(src.device)

            out = self.Encoder(src, src_key_padding_mask=src_key_padding_mask, is_causal=self.auto_regressive,
                               mask=self.causal_mask)
        else:
            out = self.Encoder(src, src_key_padding_mask=src_key_padding_mask)

        return out


    def encode_unmasked(self, src):
        out = self.Encoder(src)
        return out

# --------------------------------------------------------------------------------
# ------------         VARIAIONAL REPARAMETERIZE BLOCK       ---------------------
# --------------------------------------------------------------------------------
class LatentLayer(torch.nn.Module):
    """ Latent variable reparameterization layer

   :param input: (Tensor) Input tensor to REPARAMETERIZE [B x max_len_enc x d_model_enc]
   :return: mu, log_var, z (Tensor) [B x max_len_enc x d_model_enc]
   """

    def __init__(self, max_len, d_model, latent_dim):
        super(LatentLayer, self).__init__()

        self.fc_mu = torch.nn.Linear(int(max_len * d_model), latent_dim)
        self.fc_var = torch.nn.Linear(int(max_len * d_model), latent_dim)

    def init_weights(self, initrange=0.1):
        self.fc_mu.bias.data.zero_()
        self.fc_mu.weight.data.uniform_(-initrange, initrange)
        self.fc_var.bias.data.zero_()
        self.fc_var.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self, src: torch.Tensor):
        """ converts the input into a latent space representation

        :param src: (Tensor) Input tensor to REPARAMETERIZE [N x max_encoder_len x d_model]
        :return:  mu , logvar, z (each with dimensions [N, latent_dim_size])
        """
        result = torch.flatten(src, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # Reparameterize
        z = self.reparametrize(mu, log_var)

        return mu, log_var, z

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu

        return z


class SingleFeatureOutputLayer(torch.nn.Module):
    """ Maps the dimension of the output of a decoded sequence into to the dimension of the output

        eg. from (batch, 32, 128) to (batch, 32, 9)

        for either hits, velocities or offsets
        """

    def __init__(self, embedding_size, d_model):
        """
        Output layer of the transformer model
        :param embedding_size: size of the embedding (output dim at each time step)
        :param d_model:     size of the model         (input dim at each time step)
        """
        super(SingleFeatureOutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.fill_(0.5)
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        logits = self.Linear(decoder_out)

        return logits


class DecoderInput(torch.nn.Module):
    """ Embeds the controls and adds them to the latent space representation.
    Then the result is reshaped to [batch, max_len, d_model_dec] to be used as input to the decoder
    """

    def __init__(self, max_len, latent_dim, d_model):
        super(DecoderInput, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

        self.fc = torch.nn.Linear(int(latent_dim), int(max_len * d_model))
        self.reLU = torch.nn.ReLU()

    def init_weights(self, initrange=0.1):
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    @torch.jit.export
    def forward(self,
                latent_z: torch.Tensor):
        """
        maps the latent to decoder input

        :param latent_z: shape (batch, latent_dim)

        :return:
                projected (latent+controls) into the shape (batch, max_len, d_model_dec)
        """

        return self.reLU(self.fc.forward(latent_z)).view(-1, self.max_len, self.d_model)


import torch
def load_model(model_path, model_class, params_dict=None, is_evaluating=True, device=None):
    try:
        if device is not None:
            loaded_dict = torch.load(model_path, map_location=device)
        else:
            loaded_dict = torch.load(model_path)
    except:
        loaded_dict = torch.load(model_path, map_location=torch.device('cpu'))

    if params_dict is None:
        if 'params' in loaded_dict:
            params_dict = loaded_dict['params']
        else:
            raise Exception(f"Could not instantiate model as params_dict is not found. "
                            f"Please provide a params_dict either as a json path or as a dictionary")

    if isinstance(params_dict, str):
        import json
        with open(params_dict, 'r') as f:
            params_dict = json.load(f)

    model = model_class(params_dict)
    model.load_state_dict(loaded_dict["model_state_dict"])
    if is_evaluating:
        model.eval()

    return model
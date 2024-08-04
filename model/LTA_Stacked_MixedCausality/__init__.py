#  Copyright (c) 2024. \n Created by Behzad Haki. behzadhaki88@gmail.com

import torch
import json
import os

from model import PositionalEncoding, SingleFeatureOutputLayer


class StepEncoder(torch.nn.Module):

    def __init__(self, config):
        super(StepEncoder, self).__init__()
        self.config = config
        self.n_src_voices = self.config['n_src1_voices'] + self.config['n_src2_voices']
        self.n_src1_voices = self.config['n_src1_voices']
        self.n_src2_voices = self.config['n_src2_voices']

        if self.config['input_has_velocity']:
            self.n_feaures_per_step = 3
        else:
            self.n_feaures_per_step = 2

        in_features = self.n_feaures_per_step * self.n_src_voices
        out_features = self.config['d_model']
        print(f"StepEncoder: in_features: {in_features}, out_features: {out_features}")

        # the following stupid 16 step by step implementation is due to torchscript limitations

        self.FCN = torch.nn.Sequential(
                    torch.nn.Linear(in_features, out_features // 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_features // 2, out_features)
                )
    @torch.jit.export
    def forward(self, src: torch.Tensor):

        return self.FCN(src)

    @torch.jit.ignore
    def save(self, save_path, additional_info=None):
        """ Saves the model to the given path. The Saved pickle has all the parameters ('params' field) as well as
        the state_dict ('state_dict' field) """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config_ = dict()
        for key, value in self.config.items():
            config_[key] = value
        json.dump(config_, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': config_,
                    'additional_info': additional_info}, save_path)

    # serializes to a torchscript model
    @torch.jit.ignore
    def serialize(self, save_folder, filename=None):

        os.makedirs(save_folder, exist_ok=True)

        if filename is None:
            import datetime
            filename = f'GrooveRhythmEncoder_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        is_train = self.training
        self.eval()
        save_path = os.path.join(save_folder, filename)

        scr = torch.jit.script(self)
        # save model
        with open(save_path, "wb") as f:
            torch.jit.save(scr, f)

        if is_train:
            self.train()


# def generate_memory_mask(input_steps, output_steps):
#     assert output_steps % input_steps == 0, "output_steps must be a multiple of input_steps"
#
#     mask = torch.ones((input_steps, output_steps))
#     mask = mask * -float('inf')
#
#     ratio_ = output_steps // input_steps
#
#     for i in range(input_steps):
#         mask[i, i*ratio_:] = 0
#
#     return mask.transpose(0, 1)

def generate_memory_mask_for_K_bars_ahead_prediction(input_steps, output_steps, predict_K_bars_ahead, segment_length):
    mask = torch.ones((output_steps, int(input_steps//segment_length)), dtype=torch.float32)
    mask = mask * -float('inf')

    n_segments_per_bar = 16 // segment_length
    max_look_back_segments = predict_K_bars_ahead * n_segments_per_bar

    print(f"max_look_back_segments: {max_look_back_segments}")
    print("n_segments_per_bar: ", n_segments_per_bar)

    for i in range(output_steps):
        seg_ix = int(i / segment_length)

        if seg_ix < n_segments_per_bar:
            mask[i, :seg_ix+1] = 0
        else:
            if (seg_ix - max_look_back_segments) < n_segments_per_bar:
                mask[i, :n_segments_per_bar] = 0
            else:
                mask[i, :seg_ix - max_look_back_segments + 1] = 0
    return mask


class LTA_Stacked_MixedCausality(torch.nn.Module):

    def __init__(self, config):
        super(LTA_Stacked_MixedCausality, self).__init__()
        self.config = config
        self.max_n_steps = self.config['max_n_steps']
        self.encoder_d_model = self.config['d_model']
        self.predict_K_bars_ahead = config['predict_K_bars_ahead']
        self.n_total_voices = config['n_src1_voices'] + config['n_src2_voices']
        self.input_has_velocity = config['input_has_velocity']

        # The following will be shared every segment
        self.StepEncoder = StepEncoder(config)

        self.PositionalEncoding = PositionalEncoding(
            d_model=self.config['d_model'],
            max_len=self.config['max_n_steps'],  # maximum number of bars in a performance
            dropout=float(self.config['positional_encoding_dropout'])
        )

        norm_encoder = torch.nn.LayerNorm(self.config['d_model'])

        self.TransformerEncoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=self.config['d_model'],
                nhead=self.config['nhead'],
                dim_feedforward=self.config['d_ff'],
                dropout=float(self.config['dropout']),
                batch_first=True
            ),
            num_layers=self.config['n_layers'],
            norm=norm_encoder)

        # self.causal_mask = generate_memory_mask_for_K_bars_ahead_prediction(
        #     input_steps=self.config['max_n_steps'],
        #     output_steps=self.config['max_n_steps'],
        #     predict_K_bars_ahead=self.predict_K_bars_ahead,
        #     segment_length=1)

        self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_n_steps'])
        self.causal_mask[:32, :32] = 0  # first two bars are not causal

        self.HitOutputLayer = SingleFeatureOutputLayer(
            embedding_size=self.n_total_voices,
            d_model=self.config['d_model'],
        )

        self.VelocityOutputLayer = SingleFeatureOutputLayer(
            embedding_size=self.n_total_voices,
            d_model=self.config['d_model'],
        )

        self.OffsetOutputLayer = SingleFeatureOutputLayer(
            embedding_size=self.n_total_voices,
            d_model=self.config['d_model'],
        )

    def __getstate__(self):
        state = self.__dict__.copy()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def forward(self, shifted_tgt: torch.Tensor):

        # StepEncoder
        n_steps = int(shifted_tgt.shape[1])

        batch_size, n_steps, encoder_d_model = shifted_tgt.shape
        #shifted_tgt_reshaped = shifted_tgt.view(-1, encoder_d_model)
        shifted_tgt_reshaped = shifted_tgt.reshape(-1, encoder_d_model)

        encoded_steps_reshaped = self.StepEncoder(src=shifted_tgt_reshaped)
        encoded_steps = encoded_steps_reshaped.view(batch_size, n_steps, -1)
        #
        # encoded_steps = torch.zeros((shifted_tgt.shape[0], n_steps, self.encoder_d_model), device=shifted_tgt.device)
        # for i in range(n_steps):
        #     encoded_steps[:, i, :] = self.StepEncoder(
        #         src=shifted_tgt[:, i, :])

        # TransformerEncoder
        if self.causal_mask.device != shifted_tgt.device:
            self.causal_mask = self.causal_mask.to(shifted_tgt.device)

        src = self.PositionalEncoding(encoded_steps)
        encodings = self.TransformerEncoder.forward(
            src=src,
            is_causal=True,
            mask=self.causal_mask[:src.shape[1], :src.shape[1]])

        # decode the output groove
        h_logits = self.HitOutputLayer(encodings)
        v_logits = self.VelocityOutputLayer(encodings)
        o_logits = self.OffsetOutputLayer(encodings)

        return h_logits, v_logits, o_logits
    #forward_using_batch_data_teacher_force

    @torch.jit.ignore
    def forward_src_masked(self, shifted_tgt: torch.Tensor, teacher_forcing_ration: float=0.9):
        # used for training only

        # StepEncoder
        n_steps = int(shifted_tgt.shape[1])

        if teacher_forcing_ration < 1.0:
            indices = torch.rand(self.causal_mask.size(0))
            indices = indices > teacher_forcing_ration
            indices[:16] = False
            causal_mask = self.causal_mask.clone()
            causal_mask[:, indices] = True
        else:
            causal_mask = self.causal_mask

        batch_size, n_steps, encoder_d_model = shifted_tgt.shape
        shifted_tgt_reshaped = shifted_tgt.view(-1, encoder_d_model)
        encoded_steps_reshaped = self.StepEncoder(src=shifted_tgt_reshaped)
        encoded_steps = encoded_steps_reshaped.view(batch_size, n_steps, -1)
        #
        # encoded_steps = torch.zeros((shifted_tgt.shape[0], n_steps, self.encoder_d_model), device=shifted_tgt.device)
        # for i in range(n_steps):
        #     encoded_steps[:, i, :] = self.StepEncoder(
        #         src=shifted_tgt[:, i, :])

        # TransformerEncoder
        src = self.PositionalEncoding(encoded_steps)

        encodings = self.TransformerEncoder.forward(
            src=src,
            is_causal=True,
            mask=causal_mask[:src.shape[1], :src.shape[1]])

        # decode the output groove
        h_logits = self.HitOutputLayer(encodings)
        v_logits = self.VelocityOutputLayer(encodings)
        o_logits = self.OffsetOutputLayer(encodings)

        return h_logits, v_logits, o_logits

    @torch.jit.ignore
    def sample(self, shifted_tgt: torch.Tensor, scale_vel: float=1.0, threshold: float=0.5, use_bernulli: bool=False, temperature: float=1.0):

        h_logits, v_logits, o_logits = self.forward(
            shifted_tgt=shifted_tgt
        )

        if temperature <= 0.000001:
            temperature = 0.000001

        h_logits = h_logits / temperature
        h = torch.sigmoid(h_logits)

        # bernoulli sampling
        v = torch.clamp(((torch.tanh(v_logits) + 1.0) / 2) * scale_vel, 0.0, 1.0)
        o = torch.tanh(o_logits)

        if use_bernulli:
            h = torch.where(h < threshold, 0.0, h)
            h = torch.bernoulli(h)
        else:
            h = torch.where(h > threshold, 1.0, 0.0)

        return h, v, o, torch.cat((h, v, o), dim=-1), h_logits

    @torch.jit.ignore
    def save(self, save_path, additional_info=None):
        """ Saves the model to the given path. The Saved pickle has all the parameters ('params' field) as well as
        the state_dict ('state_dict' field) """
        if not save_path.endswith('.pth'):
            save_path += '.pth'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config_ = dict()
        for key, value in self.config.items():
            config_[key] = value
        json.dump(config_, open(save_path.replace('.pth', '.json'), 'w'))
        torch.save({'model_state_dict': self.state_dict(), 'params': config_,
                    'additional_info': additional_info}, save_path)

    # serializes to a torchscript model
    @torch.jit.ignore
    def serialize(self, save_folder, filename=None):

            os.makedirs(save_folder, exist_ok=True)

            if filename is None:
                import datetime
                filename = f'LongTermAccompaniment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            is_train = self.training
            self.eval()
            save_path = os.path.join(save_folder, filename)

            scr = torch.jit.script(self)
            # save model
            with open(save_path, "wb") as f:
                torch.jit.save(scr, f)

            if is_train:
                self.train()


if __name__ == "__main__":
    config = {
        'n_src1_voices': 1,
        'n_src2_voices': 9,
        'd_model': 64,
        'd_ff': 256,
        'dropout': 0.1,
        'n_layers': 4,
        'nhead': 4,
        'max_n_steps': 128,
        'positional_encoding_dropout': 0.1,
        'predict_K_bars_ahead': 1,
        'input_has_velocity': True
    }

    model = LTA_Stacked_MixedCausality(config)

    model.forward(torch.rand(16, 1, 30))

    model.save("misc/model_test.pth")

    model.serialize(save_folder="misc", filename="model_test.pt")
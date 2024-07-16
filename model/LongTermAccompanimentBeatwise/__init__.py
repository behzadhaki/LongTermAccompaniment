#  Copyright (c) 2024. \n Created by Behzad Haki. behzadhaki88@gmail.com

import torch
import json
import os

from model import TransformerEncoder, InputGrooveRhythmLayer, PositionalEncoding, SingleFeatureOutputLayer


class SegmentEncoder(torch.nn.Module):
    """
    An encoder only transformer
    """

    def __init__(self, config):
        """
        This is a vanilla transformer encoder that takes a sequence of hvo and encodes it into a single tensor

        In the hierarchical model, this is the encoder for the rhythm extracted from a single bar of the performance

        :param config: A dictionary containing the configuration of the model
        config must contain the following keys:
        - groove_dim: The dimension of the groove representation (3)
        - d_model: The dimension of the model
        - nhead: The number of heads in the multiheadattention models
        - dim_feedforward: The dimension of the feedforward network model
        - n_layers: The number of sub-encoder-layers in the encoder
        - dropout: The dropout value
        - velocity_dropout: The dropout value for the velocity
        - offset_dropout: The dropout value for the offset
        - positional_encoding_dropout: The dropout value for the positional encoding
        """

        super(SegmentEncoder, self).__init__()
        self.n_feaures_per_step = 3
        self.config = config['SegmentEncoder'] if 'SegmentEncoder' in config else config
        self.n_src_voices = self.config['n_src1_voices'] + self.config['n_src2_voices']
        self.n_src1_voices = self.config['n_src1_voices']
        self.n_src2_voices = self.config['n_src2_voices']
        self.has_velocity = self.config['has_velocity'] # whether the input has velocity <--- Not used internally, just for access through serialized model (make sure the data are correctly set to zero externally)
        self.has_offset = self.config['has_offset'] # whether the input has offset <--- Not used here, just for access through serialized model
        self.steps_per_segment = self.config['steps_per_segment']

        # use linear layer to project the output of the encoder to mix self.steps_per_segment into a single tensor
        self.FCN = torch.nn.Linear(
            in_features=self.steps_per_segment * self.n_src_voices * self.n_feaures_per_step,
            out_features=self.config['d_model']
        )
        self.FCN.bias.data.zero_()
        self.FCN.weight.data.uniform_(-0.1, 0.1)

    @torch.jit.export
    def forward(self, src: torch.Tensor):
        """
        Encodes the input sequence through the encoder and predicts the latent space

        returns the last time step of the memory tensor

        :param src: [N, 16, 3]
        :return: memory: [N, d_model]

        """
        return self.FCN(src.view(src.shape[0], -1))

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


class PerformanceEncoder(torch.nn.Module):

        def __init__(self, config):
            super(PerformanceEncoder, self).__init__()

            self.config = config['PerformanceEncoder'] if 'PerformanceEncoder' in config else config

            self.PositionalEncoding = PositionalEncoding(
                d_model=self.config['d_model'],
                max_len=self.config['max_n_segments'],    # maximum number of bars in a performance
                dropout=float(self.config['positional_encoding_dropout'])
            )

            norm_encoder = torch.nn.LayerNorm(self.config['d_model'])
            self.auto_regressive = True

            self.Encoder = torch.nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    d_model=self.config['d_model'],
                    nhead=self.config['nhead'],
                    dim_feedforward=self.config['dim_feedforward'],
                    dropout=float(self.config['dropout']),
                    batch_first=True
                ),
                num_layers=self.config['n_layers'],
                norm=norm_encoder)

            self.max_n_segments = self.config['max_n_segments']

            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_n_segments'])


            # self.Encoder = TransformerEncoder(
            #     d_model=self.config['d_model'],
            #     nhead=self.config['nhead'],
            #     dim_feedforward=self.config['dim_feedforward'],
            #     num_encoder_layers=self.config['n_layers'],
            #     dropout=float(self.config['dropout']),
            #     auto_regressive=True,
            #     max_len=self.config['max_n_segments']
            # )



        @torch.jit.export
        def forward(self, src_rhythm_encodings: torch.Tensor):
            src = self.PositionalEncoding(src_rhythm_encodings)
            if self.auto_regressive:
                if self.causal_mask.device != src.device:
                    self.causal_mask = self.causal_mask.to(src.device)

                perf_encodings = self.Encoder.forward(
                    src=src,
                    is_causal=self.auto_regressive,
                    mask=self.causal_mask[:src.shape[1], :src.shape[1]])
            else:
                perf_encodings = self.Encoder(src)

            return perf_encodings

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
                filename = f'PerformanceEncoder_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            is_train = self.training
            self.eval()
            save_path = os.path.join(save_folder, filename)

            scr = torch.jit.script(self)
            # save model
            with open(save_path, "wb") as f:
                torch.jit.save(scr, f)

            if is_train:
                self.train()


def generate_memory_mask(input_steps, output_steps):
    assert output_steps % input_steps == 0, "output_steps must be a multiple of input_steps"

    mask = torch.ones((input_steps, output_steps))
    mask = mask * -float('inf')

    ratio_ = output_steps // input_steps

    for i in range(input_steps):
        mask[i, i*ratio_:] = 0

    return mask.transpose(0, 1)


class DrumDecoder(torch.nn.Module):

        def __init__(self, config):
            super(DrumDecoder, self).__init__()

            self.config = config['DrumDecoder'] if 'DrumDecoder' in config else config
            num_features = 3

            self.InputLayerEncoder = InputGrooveRhythmLayer(
                embedding_size=self.config['n_tgt_voices']*num_features,
                d_model=self.config['d_model'],
                max_len=self.config['max_steps'],
                velocity_dropout=float(self.config['velocity_dropout']),
                offset_dropout=float(self.config['offset_dropout']),
                positional_encoding_dropout=float(self.config['dropout'])
            )

            self.Decoder = torch.nn.TransformerDecoder(
                decoder_layer=torch.nn.TransformerDecoderLayer(
                    d_model=self.config['d_model'],
                    nhead=self.config['nhead'],
                    dim_feedforward=self.config['dim_feedforward'],
                    dropout=float(self.config['dropout']),
                    batch_first=True),
                num_layers=self.config['n_layers'],
                norm=torch.nn.LayerNorm(self.config['d_model'])
            )

            self.HitOutputLayer = SingleFeatureOutputLayer(
                embedding_size=self.config['n_tgt_voices'],
                d_model=self.config['d_model'],
            )

            self.VelocityOutputLayer = SingleFeatureOutputLayer(
                embedding_size=self.config['n_tgt_voices'],
                d_model=self.config['d_model'],
            )

            self.OffsetOutputLayer = SingleFeatureOutputLayer(
                embedding_size=self.config['n_tgt_voices'],
                d_model=self.config['d_model'],
            )

            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_steps'])

            self.input_segments = config['PerformanceEncoder']['max_n_segments']
            self.memory_mask = generate_memory_mask(self.input_segments, self.config['max_steps'])

            self.performance_encoder_input_steps = config['SegmentEncoder']

        @torch.jit.export
        def forward(self, tgt: torch.Tensor, memory: torch.Tensor):
            out_with_position, hit, hvo_projection = self.InputLayerEncoder(hvo=tgt)

            # replace True to -inf, False to 0

            if self.causal_mask.device != out_with_position.device:
                self.causal_mask = self.causal_mask.to(out_with_position.device)

            if self.memory_mask.device != memory.device:
                self.memory_mask = self.memory_mask.to(memory.device)

            output = self.Decoder(
                tgt=out_with_position,
                memory=memory,
                tgt_mask=self.causal_mask[:tgt.shape[1], :tgt.shape[1]],
                memory_mask=self.memory_mask[:tgt.shape[1], :memory.shape[1]]
            )

            return self.HitOutputLayer(output), self.VelocityOutputLayer(output), self.OffsetOutputLayer(output)


            # return self.Decoder(src_rhythm_encodings, memory_key_padding_mask=memory_key_padding_mask, tgt_mask=self.causal_mask)

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
                    filename = f'Performance2GrooveDecoder_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                is_train = self.training
                self.eval()
                save_path = os.path.join(save_folder, filename)

                scr = torch.jit.script(self)
                # save model
                with open(save_path, "wb") as f:
                    torch.jit.save(scr, f)

                if is_train:
                    self.train()


class LongTermAccompanimentBeatwise(torch.nn.Module):

    def __init__(self, config):
        super(LongTermAccompanimentBeatwise, self).__init__()
        self.config = config
        self.max_n_segments = self.config['PerformanceEncoder']['max_n_segments']
        self.encoder_d_model = self.config['PerformanceEncoder']['d_model']
        self.n_steps_per_segment = self.config['SegmentEncoder']['steps_per_segment']

        # The following will be shared every segment
        self.SegmentEncoder = SegmentEncoder(config)

        self.PerformanceEncoder = PerformanceEncoder(config)
        self.DrumDecoder = DrumDecoder(config)

        # inference only utils
        self.encoded_segments = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.performance_memory = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.num_bars_encoded_so_far = 0
        self.max_tgt_steps = self.config['DrumDecoder']['max_steps']
        self.num_tgt_voices = self.config['DrumDecoder']['n_tgt_voices']

    @torch.jit.ignore
    def freeze_decoder(self):
        # freeze decoder
        self.DrumDecoder.requires_grad_(False)

    @torch.jit.ignore
    def unfreeze_decoder(self):
        self.DrumDecoder.requires_grad_(True)

    def forward(self, src: torch.Tensor, shifted_tgt: torch.Tensor):

        # SegmentEncoder
        n_segments = int(src.shape[1] // self.n_steps_per_segment)

        encoded_segments = torch.zeros((src.shape[0], n_segments, self.encoder_d_model), device=src.device)
        for i in range(n_segments):
            encoded_segments[:, i, :] = self.SegmentEncoder.forward(
                src=src[:, i * self.n_steps_per_segment: (i + 1) * self.n_steps_per_segment, :])

        # PerformanceEncoder
        perf_encodings = self.PerformanceEncoder.forward(
            src_rhythm_encodings=encoded_segments)

        # decode the output groove
        h_logits, v_logits, o_logits = self.DrumDecoder.forward(
            tgt=shifted_tgt,
            memory=perf_encodings)

        return h_logits, v_logits, o_logits

    @torch.jit.ignore
    def exclude_decoder_from_backprop(self):
        self.DrumDecoder.requires_grad_(False)

    @torch.jit.ignore
    def sample(self, src: torch.Tensor, tgt: torch.Tensor, scale_vel: float=1.0, threshold: float=0.5, use_bernulli: bool=False, temperature: float=1.0):


        h_logits, v_logits, o_logits = self.forward(
            src=src,
            shifted_tgt=tgt
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

    @torch.jit.export
    def stack_two_hvos(self, hvo1: torch.Tensor, hvo2: torch.Tensor):
        if len(hvo1.shape) > 2 or len(hvo2.shape) > 2:
            if hvo1.shape[0] != 1 or hvo2.shape[0] != 1:
                print("Only batch size of 1 is supported for the input hvo tensors")
            hvo1 = hvo1.squeeze(0)
            hvo2 = hvo2.squeeze(0)
            has_batch_dim = True
        else:
            has_batch_dim = False

        n_voices_1 = self.SegmentEncoder.n_src1_voices
        n_voices_2 = self.SegmentEncoder.n_src2_voices

        h1 = hvo1[:, :n_voices_1]
        v1 = hvo1[:, n_voices_1:2 * n_voices_1]
        o1 = hvo1[:, 2 * n_voices_1:]
        h2 = hvo2[:, :n_voices_2]
        v2 = hvo2[:, n_voices_2:2 * n_voices_2]
        o2 = hvo2[:, 2 * n_voices_2:]
        return torch.hstack([h1, h2, v1, v2, o1, o2]).unsqueeze(0) if has_batch_dim else torch.hstack(
            [h1, h2, v1, v2, o1, o2])

    @torch.jit.export
    def add_single_bar_of_instrumental_pair(self, src_instr_hvo: torch.Tensor, tgt_instr_hvo: torch.Tensor):
        """
        Encodes a single bar of the performance
        """

        if len(src_instr_hvo.shape) == 2:
            src_instr_hvo = src_instr_hvo.unsqueeze(0)

        if len(tgt_instr_hvo.shape) == 2:
            tgt_instr_hvo = tgt_instr_hvo.unsqueeze(0)

        if (src_instr_hvo.shape[1] != 16) or (tgt_instr_hvo.shape[1] != 16):
            print(f'Error: The input instrumental groove and the drums must have 16 steps (i.e. 1 bar). The input has {src_instr_hvo.shape[1]} steps')

        if (src_instr_hvo.shape[-1] + tgt_instr_hvo.shape[-1]) != (self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src_voices):
            print(f'Error: The input instrumental groove and the drums must have a total of {self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src_voices} features. The input has {src_instr_hvo.shape[-1] + tgt_instr_hvo.shape[-1]} features')

        with torch.no_grad():
            # make sure there is at least one empty bar to add new input pairs
            if self.num_bars_encoded_so_far == self.max_n_segments:
                self.shift_by_n_bars(
                    n_bars=1,
                    adapt_num_bars_encoded_so_far=True)

            self.encoded_segments[:, self.num_bars_encoded_so_far, :] = self.SegmentEncoder.forward(
                src=self.stack_two_hvos(src_instr_hvo, tgt_instr_hvo))

            self.performance_memory = self.PerformanceEncoder.forward(
                src_rhythm_encodings=self.encoded_segments)

    @torch.jit.export
    def encode_src_inst_without_tgt(self, src_instr_hvo: torch.Tensor):
        if src_instr_hvo.shape[-1] != self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src1_voices:
            print(f'Error: The input instrumental groove must have {self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src1_voices} features. The input has {src_instr_hvo.shape[-1]} features')

        if len(src_instr_hvo.shape) == 2:
            src_instr_hvo = src_instr_hvo.unsqueeze(0)

        if src_instr_hvo.shape[1] % 16 != 0:
            print(f'Error: The input instrumental groove must have a multiple of 16 steps. The input has {src_instr_hvo.shape[1]} steps')

        tgt_instr_hvo = torch.zeros((src_instr_hvo.shape[0], src_instr_hvo.shape[1], self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src2_voices))

        self.encode_varying_length_performance(src_instr_hvo, tgt_instr_hvo)

    @torch.jit.export
    def encode_tgt_inst_without_src(self, tgt_instr_hvo: torch.Tensor):
        if tgt_instr_hvo.shape[-1] != self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src2_voices:
            print(f'Error: The input instrumental groove must have {self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src2_voices} features. The input has {tgt_instr_hvo.shape[-1]} features')

        if len(tgt_instr_hvo.shape) == 2:
            tgt_instr_hvo = tgt_instr_hvo.unsqueeze(0)

        if tgt_instr_hvo.shape[1] % 16 != 0:
            print(f'Error: The input instrumental groove must have a multiple of 16 steps. The input has {tgt_instr_hvo.shape[1]} steps')

        src_instr_hvo = torch.zeros((tgt_instr_hvo.shape[0], tgt_instr_hvo.shape[1], self.SegmentEncoder.n_feaures_per_step * self.SegmentEncoder.n_src1_voices))

        self.encode_varying_length_performance(src_instr_hvo, tgt_instr_hvo)


    @torch.jit.export
    def encode_varying_length_performance(self, src_instr_hvo: torch.Tensor, tgt_instr_hvo: torch.Tensor):
        if len(src_instr_hvo.shape) == 2:
            src_instr_hvo = src_instr_hvo.unsqueeze(0)

        if len(tgt_instr_hvo.shape) == 2:
            tgt_instr_hvo = tgt_instr_hvo.unsqueeze(0)

        if (src_instr_hvo.shape[1] % 16 != 0) or (tgt_instr_hvo.shape[1] % 16 != 0):
            print(f'Error: The input instrumental groove and the drums must have a multiple of 16 steps.')

        n_bars = int(src_instr_hvo.shape[1] // 16)

        for i in range(n_bars):
            self.add_single_bar_of_instrumental_pair(src_instr_hvo[:, i*16: (i+1)*16, :], tgt_instr_hvo[:, i*16: (i+1)*16, :])
            self.moved_to_next_bar()

    @torch.jit.export
    def moved_to_next_bar(self):
        self.num_bars_encoded_so_far += 1

    @torch.jit.export
    def get_next_2_bars(self, threshold: float= 0.5):
        with torch.no_grad():
            hits_shifted = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
            vels_shifted = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
            offsets_shifted = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))

            current_step_hits = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
            v = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
            o = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))

            for i in range(self.max_tgt_steps):
                h_logits, v_logits, o_logits = self.DrumDecoder.forward(
                    tgt=torch.cat((hits_shifted, vels_shifted, offsets_shifted), dim=-1),
                    memory=self.performance_memory)

                h = torch.sigmoid(h_logits)
                v = (torch.tanh(v_logits) + 1.0) / 2
                o = torch.tanh(o_logits)

                current_step_hits = torch.where(h > threshold, 1, 0)
                v = v * current_step_hits
                o = o * current_step_hits

                if i < self.max_tgt_steps - 1:
                    hits_shifted[:, i+1, :] = current_step_hits[:, i, :]
                    vels_shifted[:, i+1, :] = v[:, i, :]
                    offsets_shifted[:, i+1, :] = o[:, i, :]

        return current_step_hits, v, o, torch.concat((current_step_hits, v, o), dim=-1)


    @torch.jit.export
    def reset(self):
        self.encoded_segments = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.num_bars_encoded_so_far = 0

    @torch.jit.export
    def shift_by_n_bars(self, n_bars: int, adapt_num_bars_encoded_so_far: bool=True):
        # shift left by n_bars and fill the rest with zeros
        self.encoded_segments[:, :self.max_n_segments - n_bars, :] = self.encoded_segments[:, n_bars:, :].clone()
        self.encoded_segments[:, self.max_n_segments - n_bars:, :] = 0

        if adapt_num_bars_encoded_so_far:
            self.num_bars_encoded_so_far -= n_bars

    @torch.jit.export
    def encode_next_performed_groove_bar(self, src: torch.Tensor):
        if (len(src.shape) == 2):
            src = src.unsqueeze(0)

        if (src.shape[1] != 16):
            print(f'Error: The input groove must have 16 steps. The input has {src.shape[1]} steps')

        self.encoded_segments[:, self.num_bars_encoded_so_far, :] = self.SegmentEncoder.forward(
            src=src[:, :16, :])

        self.performance_memory = self.PerformanceEncoder.forward(
            src_rhythm_encodings=self.encoded_segments)

    @torch.jit.export
    def get_upcoming_2bars(self, sampling_thresh: float = 0.5):

        hits_shifted = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
        vels_shifted = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
        offsets_shifted = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))

        for i in range(self.max_tgt_steps):
            h_logits, v_logits, o_logits = self.DrumDecoder.forward(
                tgt=torch.cat((hits_shifted, vels_shifted, offsets_shifted), dim=-1),
                memory=self.performance_memory)

            step_h = torch.sigmoid(h_logits[:, i, :])
            # set over the threshold to 1
            step_v = (torch.tanh(v_logits[:, i, :]) + 1.0) / 2
            step_o = torch.tanh(o_logits[:, i, :])

            current_step_hits = torch.where(step_h > sampling_thresh, 1, 0)
            current_step_velocities = step_v * current_step_hits
            current_step_offsets =  step_o * current_step_hits

            hits_shifted[:, i, :] = current_step_hits
            vels_shifted[:, i, :] = current_step_velocities
            offsets_shifted[:, i, :] = current_step_offsets

        return hits_shifted, vels_shifted, offsets_shifted, torch.concat((hits_shifted, vels_shifted, offsets_shifted), dim=-1)

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


if __name__ == '__main__':
    steps_per_segment = 4
    max_n_segments = 32 * 4
    max_bars = int(max_n_segments * steps_per_segment / 16)

    config = {
        'SegmentEncoder': {
            'd_model': 512,
            'dim_feedforward': 2048,
            'n_layers': 6,
            'nhead': 8,
            'n_src1_voices': 1,
            'n_src2_voices': 0,
            'steps_per_segment': steps_per_segment,
            'has_velocity': True,
            'has_offset': True,
            'dropout': 0.1,
            'velocity_dropout': 0.1,
            'offset_dropout': 0.1,
            'positional_encoding_dropout': 0.1,
        },
        'PerformanceEncoder': {
            'd_model': 512,
            'dim_feedforward': 2048,
            'n_layers': 6,
            'nhead': 8,
            'max_n_segments': max_n_segments,  # maximum number of bars in a performance
            'dropout': 0.1,
            'positional_encoding_dropout': 0.1
        },
        'DrumDecoder': {
            'd_model': 512,
            'dim_feedforward': 2048,
            'n_layers': 8,
            'nhead': 4,
            'n_tgt_voices': 9,
            'max_steps': int(max_bars * 16),
            'dropout': 0.1,
            'velocity_dropout': 0.1,
            'offset_dropout': 0.1,
            'positional_encoding_dropout': 0.1,
        }
    }


    # LongTermAccompanimentHierarchical
    model = LongTermAccompanimentBeatwise(config)


    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset
    from data import PairedLTADatasetV2

    # load dataset as torch.utils.data.Dataset
    training_dataset = PairedLTADatasetV2(
        input_inst_dataset_bz2_filepath="data/lmd/data_bass_groove_test.bz2",
        output_inst_dataset_bz2_filepath="data/lmd/data_drums_full_unsplit.bz2",
        shift_tgt_by_n_steps=16,
        max_input_bars=32,
        hop_n_bars=8,
        input_has_velocity=True,
        input_has_offsets=True
    )

    # access a batch
    i1, i2, i12, i2_shifted = training_dataset[:2]


    def create_src_mask(n_bars, max_n_bars):
        # masked items are the ones noted as True

        batch_size = n_bars.shape[0]
        mask = torch.zeros((batch_size, max_n_bars), dtype=torch.bool)
        for i in range(batch_size):
            mask[i, n_bars[i]:] = 1
        return mask

    mask = create_src_mask(torch.tensor([12]*i12.shape[0]), config['PerformanceEncoder']['max_n_segments'])

    h_logits, v_log, o_log = model.forward(
        src=i1[:, :64, :],
        shifted_tgt=i2_shifted[:, :32, :]
    )
    model.eval()

    model.serialize(save_folder='misc/lta_segment_encoder', filename='test.pt')

    #     h_logits, v_log, o_log = model.forward(
    #         src=torch.rand(b_size, 16 * config['PerformanceEncoder']['max_n_segments'], n_features * config['GrooveEncoder']['n_src_voices']),
    #         src_key_padding_and_memory_mask=mask,
    #         shifted_tgt=torch.rand(b_size, config['DrumDecoder']['max_steps'],
    #                        3 * config['DrumDecoder']['n_tgt_voices']))
    #     print("Time taken for inference: LTA.forward()", time() - start)
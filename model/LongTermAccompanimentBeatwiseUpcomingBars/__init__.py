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

            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_n_segments'], dtype=torch.bool)


            # self.Encoder = TransformerEncoder(
            #     d_model=self.config['d_model'],
            #     nhead=self.config['nhead'],
            #     dim_feedforward=self.config['dim_feedforward'],
            #     num_encoder_layers=self.config['n_layers'],
            #     dropout=float(self.config['dropout']),
            #     auto_regressive=True,
            #     max_len=self.config['max_n_segments']
            # )

        def __getstate__(self):
            state = self.__dict__.copy()
            # drop the tensors
            del state['causal_mask']

            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_n_segments'], dtype=torch.bool)

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

def generate_memory_mask_for_K_bars_ahead_prediction(input_steps, output_steps, predict_K_bars_ahead, segment_length):
    mask = torch.ones((output_steps, int(input_steps//segment_length)), dtype=torch.bool)
    # mask = mask * -float('inf')

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

            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_steps'], dtype=torch.bool)

            self.input_segments = config['PerformanceEncoder']['max_n_segments']

            self.memory_mask = generate_memory_mask_for_K_bars_ahead_prediction(
                input_steps=self.input_segments * config['SegmentEncoder']['steps_per_segment'],
                output_steps=self.config['max_steps'],
                predict_K_bars_ahead=config['predict_K_bars_ahead'],
                segment_length=config['SegmentEncoder']['steps_per_segment'])

            self.performance_encoder_input_steps = config['SegmentEncoder']

        def __getstate__(self):
            state = self.__dict__.copy()
            # drop the tensors
            del state['causal_mask']
            del state['memory_mask']

            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_steps'], dtype=torch.bool)
            self.memory_mask = generate_memory_mask_for_K_bars_ahead_prediction(
                input_steps=self.input_segments * self.performance_encoder_input_steps['steps_per_segment'],
                output_steps=self.config['max_steps'],
                predict_K_bars_ahead=self.config['predict_K_bars_ahead'],
                segment_length=self.performance_encoder_input_steps['steps_per_segment'])

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


class LongTermAccompanimentBeatwiseUpcomingBars(torch.nn.Module):

    def __init__(self, config):
        super(LongTermAccompanimentBeatwiseUpcomingBars, self).__init__()
        self.config = config
        self.max_n_segments = self.config['PerformanceEncoder']['max_n_segments']
        self.encoder_d_model = self.config['PerformanceEncoder']['d_model']
        self.n_steps_per_segment = self.config['SegmentEncoder']['steps_per_segment']
        self.predict_K_bars_ahead = config['predict_K_bars_ahead']

        # The following will be shared every segment
        self.SegmentEncoder = SegmentEncoder(config)

        self.PerformanceEncoder = PerformanceEncoder(config)
        self.DrumDecoder = DrumDecoder(config)

        # inference only utils
        self.encoded_segments = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.performance_memory = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.num_segments_encoded_so_far = 0
        self.shifted_tgt = torch.zeros((1, self.config['DrumDecoder']['max_steps']+1, 3 * self.config['DrumDecoder']['n_tgt_voices']))
        self.generations = torch.zeros((1, self.config['DrumDecoder']['max_steps'], 3 * self.config['DrumDecoder']['n_tgt_voices']))

        self.max_tgt_steps = self.config['DrumDecoder']['max_steps']
        self.num_tgt_voices = self.config['DrumDecoder']['n_tgt_voices']

        # new variables post training (might be missing in the serialized/saved models)
        self.input_has_velocity = self.SegmentEncoder.has_velocity
        self.decoder_input_has_velocity = self.DrumDecoder.config['has_velocity']

        # sampling parameters
        self.kick_is_muted = False
        self.snare_is_muted = False
        self.hihat_is_muted = False
        self.tom_is_muted = False
        self.crash_is_muted = False
        self.ride_is_muted = False
        self.temperature = 1.0
        self.primed_for_N_segments = 0
        self.num_generated_steps_available = 0
        self.total_shifts = 0
    def __getstate__(self):
        state = self.__dict__.copy()
        # drop the tensors
        del state['encoded_segments']
        del state['performance_memory']
        del state['shifted_tgt']
        del state['generations']
        del state['per_voice_thresholds']
        del state['num_generated_steps_available']
        del state['total_shifts']

        # sampling parameters
        state['kick_is_muted'] =  False
        state['snare_is_muted'] = False
        state['hihat_is_muted'] = False
        state['tom_is_muted'] = False
        state['crash_is_muted'] = False
        state['ride_is_muted'] = False
        state['temperature'] = 1.0


        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'input_has_velocity' not in self.__dict__:
            self.input_has_velocity = self.SegmentEncoder.has_velocity
        if 'decoder_input_has_velocity' not in self.__dict__:
            self.decoder_input_has_velocity = self.DrumDecoder.config['has_velocity']
        if 'num_beats_encoded_so_far' in self.__dict__:
            self.num_segments_encoded_so_far = self.num_segments_encoded_so_far
            del self.num_segments_encoded_so_far
        # create the tensors
        self.encoded_segments = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.performance_memory = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.shifted_tgt = torch.zeros((1, self.config['DrumDecoder']['max_steps']+32+1, 3 * self.config['DrumDecoder']['n_tgt_voices']))
        self.generations = torch.zeros((1, self.config['DrumDecoder']['max_steps']+32, 3 * self.config['DrumDecoder']['n_tgt_voices']))
        self.num_generated_steps_available = 0
        self.total_shifts = 0

    @torch.jit.export
    def mute_voice(self, voice: str):
        if 'kick' in voice.lower():
            self.kick_is_muted = True
        elif 'snare' in voice.lower():
            self.snare_is_muted = True
        elif 'hat' in voice.lower():
            self.hihat_is_muted = True
        elif 'tom' in voice.lower():
            self.tom_is_muted = True
        elif 'crash' in voice.lower():
            self.crash_is_muted = True
        elif 'ride' in voice.lower():
            self.ride_is_muted = True

    @torch.jit.export
    def unmute_voice(self, voice: str):
        if 'kick' in voice.lower():
            self.kick_is_muted = False
        elif 'snare' in voice.lower():
            self.snare_is_muted = False
        elif 'hat' in voice.lower():
            self.hihat_is_muted = False
        elif 'tom' in voice.lower():
            self.tom_is_muted = False
        elif 'crash' in voice.lower():
            self.crash_is_muted = False
        elif 'ride' in voice.lower():
            self.ride_is_muted = False

    @torch.jit.export
    def generate_random_pattern(self, n_bars: int=2, threshold: float=0.05,
                                temperature: float=1.0, no_kick: bool=False, no_snare: bool=False, no_hihat: bool=False,
                                no_toms: bool=False, no_crash: bool=False, no_ride: bool=False):

        # generate random segment embeddings
        rand_segment_embs = torch.rand((1, n_bars, self.encoder_d_model))
        performance_memory = self.PerformanceEncoder.forward(rand_segment_embs)

        if temperature <= 0.000001:
            temperature = 0.000001

        n_iters = n_bars * 16

        hvo_shifted = torch.zeros((1, n_iters+1, self.num_tgt_voices * 3))
        generated_hvo = torch.zeros((1, n_iters, self.num_tgt_voices * 3))

        for i in range(n_iters):

            h_logits, v_logits, o_logits = self.DrumDecoder.forward(
                tgt=hvo_shifted[:, :i+1, :],
                memory=performance_memory
            )

            h_logits = h_logits / temperature
            h_ = torch.sigmoid(h_logits[:, i, :])

            # bernoulli sampling
            v_ = torch.clamp(((torch.tanh(v_logits[:, i, :]) + 1.0) / 2), 0.0, 1.0)
            o_ = torch.tanh(o_logits[:, i, :])

            h_ = torch.where(h_ < threshold, 0.0, h_)
            h_ = torch.bernoulli(h_)

            if no_kick:
                h_[:, :, 0::9] = 0
            if no_snare:
                h_[:, :, 1::9] = 0
            if no_hihat:
                h_[:, :, 2::9] = 0
                h_[:, :, 3::9] = 0
            if no_toms:
                h_[:, :, 4::9] = 0
                h_[:, :, 5::9] = 0
                h_[:, :, 6::9] = 0
            if no_crash:
                h_[:, :, 7::9] = 0
            if no_ride:
                h_[:, :, 8::9] = 0

            generated_hvo[:, i, :] = torch.cat((h_, v_, o_), dim=-1)

            if not self.decoder_input_has_velocity:
                v_ = torch.ones_like(v_) * 0

            hvo_shifted[:, i+1, :] = torch.cat((h_, v_, o_), dim=-1)

        return generated_hvo

    @torch.jit.export
    def reset_all(self):
        self.encoded_segments = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.performance_memory = torch.zeros((1, self.max_n_segments, self.encoder_d_model))
        self.shifted_tgt *= 0
        self.generations *= 0
        self.num_segments_encoded_so_far = 0
        self.num_generated_steps_available = 0
        self.total_shifts = 0


    @torch.jit.export
    def shift_left_by(self, n_segments: int, ensure_start_is_bar_boundary: bool=True):

        if ensure_start_is_bar_boundary:
            n_segments_per_bar = 16 // self.n_steps_per_segment
            # make large enough to ensure the start is a bar boundary
            if n_segments % n_segments_per_bar != 0:
                n_segments = int(n_segments//n_segments_per_bar + 1) * n_segments_per_bar

        self.total_shifts += n_segments * self.n_steps_per_segment

        self.encoded_segments = torch.roll(self.encoded_segments, shifts=-n_segments, dims=1)
        self.encoded_segments[:, self.num_segments_encoded_so_far - n_segments:, :] = 0
        self.performance_memory = torch.roll(self.performance_memory, shifts=-n_segments, dims=1)
        self.performance_memory[:, self.num_segments_encoded_so_far - n_segments:, :] = 0
        self.num_segments_encoded_so_far -= n_segments
        self.shifted_tgt = torch.roll(self.shifted_tgt, shifts=-n_segments * self.n_steps_per_segment, dims=1)
        self.shifted_tgt[:, self.max_tgt_steps - n_segments * self.n_steps_per_segment:, :] = 0
        self.shifted_tgt[:, 0, :] = 0 # start token
        self.generations = torch.roll(self.generations, shifts=-n_segments * self.n_steps_per_segment, dims=1)
        self.generations[:, self.max_tgt_steps - n_segments * self.n_steps_per_segment:, :] = 0
        self.num_generated_steps_available = max(0, self.num_generated_steps_available - n_segments * self.n_steps_per_segment)

    @torch.jit.export
    def encode_input_performance(self, hvo_: torch.Tensor):
        hvo = hvo_.clone()
        num_segments_per_bar = 16 // self.n_steps_per_segment

        if len(hvo.shape) == 2:
            # add batch dimension
            hvo = hvo.unsqueeze(0)

        if not self.input_has_velocity:
            hvo[:, :, 1] *= 0


        if hvo.shape[1] % self.n_steps_per_segment != 0:
            print("The input performance must be a multiple of the number of steps per segment")
            print("Zero padding will be added to the end of the performance to make it a multiple of the number of steps per segment")
            n_pad = self.n_steps_per_segment - (hvo.shape[1] % self.n_steps_per_segment)
            hvo = torch.cat((hvo, torch.zeros((hvo.shape[0], n_pad, hvo.shape[2]), device=hvo.device)), dim=1)

        n_segments = int(hvo.shape[1] / self.n_steps_per_segment)

        if n_segments + self.num_segments_encoded_so_far + num_segments_per_bar * self.predict_K_bars_ahead > self.max_n_segments: # add an extra bar for shifting
            shift_by_amount = 8 * 16 // self.n_steps_per_segment # add an extra 8bars for shifting
            print("SHIFTING THE MEMORY TO THE LEFT BY", shift_by_amount, "SEGMENTS")
            self.shift_left_by(shift_by_amount, ensure_start_is_bar_boundary=True)
            # self.performance_memory = self.performance_memory[:, shift_by:, :]

        for i in range(n_segments):
            self.encoded_segments[:, self.num_segments_encoded_so_far + i, :] = self.SegmentEncoder.forward(
                src=hvo[:, i * self.n_steps_per_segment: (i + 1) * self.n_steps_per_segment, :])

        self.num_segments_encoded_so_far += n_segments

        self.performance_memory[:, :self.num_segments_encoded_so_far, :] = self.PerformanceEncoder.forward(
            src_rhythm_encodings=self.encoded_segments[:, :self.num_segments_encoded_so_far, :])


    @torch.jit.export
    def prime_with_drums(self, hvo: torch.Tensor):
        self.reset_all()
        n_steps = hvo.shape[1]
        if n_steps % 16 != 0:
            print("The priming pattern must be flushed to a bar boundary - will be truncated to the last bar boundary")
            n_steps = int(n_steps // 16) * 16
            hvo = hvo[:, :n_steps, :]

        if (n_steps//self.n_steps_per_segment) > self.max_n_segments:
            print("Pattern is larger than the allowed maximum - will be truncated to the maximum allowed")
            hvo = hvo[:, :self.max_n_segments * self.n_steps_per_segment, :]

        self.primed_for_N_segments = n_steps // self.n_steps_per_segment

        self.generations[:, :n_steps, :] = hvo.clone()
        self.shifted_tgt[:, 1:n_steps+1, :] = hvo.clone()

    @torch.jit.export
    def get_segments_encoded_so_far(self):
        return self.num_segments_encoded_so_far

    @torch.jit.export
    def get_num_look_back_segments(self):
        return self.predict_K_bars_ahead



    @torch.jit.export
    def predict_first_bar(self, threshold: float=0.05):

        n_segments_per_bar = 16 // self.n_steps_per_segment

        if self.primed_for_N_segments > 0:
            print("Nothing to generate - already primed for", self.primed_for_N_segments, "segments - returning first bar of the primed pattern")
            return self.generations[:, :16, :]

        # should have all necessary segments for the first bar
        if self.num_segments_encoded_so_far < n_segments_per_bar:
            print("Too soon to generate a pattern - please provide more input - So far only", self.num_segments_encoded_so_far, "segments have been encoded")
            return torch.zeros((1, 16, 3 * self.num_tgt_voices))

        for i in range(16):

                h_logits, v_logits, o_logits = self.DrumDecoder.forward(
                    tgt=self.shifted_tgt[:, :i+1, :],
                    memory=self.performance_memory[:, :n_segments_per_bar, :]   # provide all segments encoded so far
                )

                h_logits = h_logits[:, i, :] / self.temperature
                h = torch.sigmoid(h_logits)

                # bernoulli sampling
                v = torch.clamp(((torch.tanh(v_logits[:, i, :]) + 1.0) / 2), 0.0, 1.0)
                o = torch.tanh(o_logits[:, i, :])

                h = torch.where(h < threshold, 0.0, h)
                h = torch.bernoulli(h)

                if self.kick_is_muted:
                    h[:, 0::9] = 0
                if self.snare_is_muted:
                    h[:, 1::9] = 0
                if self.hihat_is_muted:
                    h[:, 2::9] = 0
                    h[:, 3::9] = 0
                if self.tom_is_muted:
                    h[:, 4::9] = 0
                    h[:, 5::9] = 0
                    h[:, 6::9] = 0
                if self.crash_is_muted:
                    h[:, 7::9] = 0
                if self.ride_is_muted:
                    h[:, 8::9] = 0

                self.generations[:, i, :] = torch.cat((h, v, o), dim=-1)

                if self.decoder_input_has_velocity:
                    self.shifted_tgt[:, i+1, :] = torch.cat((h, v, o), dim=-1)
                else:
                    self.shifted_tgt[:, i+1, :] = torch.cat((h, torch.zeros_like(v) * 0, o), dim=-1)

        return self.generations[:, :16, :]

    @torch.jit.export
    def predict_n_steps_without_safety_checks(self, start_: int, n_steps_:int, threshold: float = 0.05):
        """
        only used internally in the following method! DO NOT USE in DEPLOYMENT
        """
        for i in range(start_, start_ + n_steps_):

            h_logits, v_logits, o_logits = self.DrumDecoder.forward(
                tgt=self.shifted_tgt[:, :i + 1, :],
                memory=self.performance_memory[:, :self.num_segments_encoded_so_far, :]
            )

            h_logits = h_logits[:, i, :] / self.temperature
            h = torch.sigmoid(h_logits)

            # bernoulli sampling
            v = torch.clamp(((torch.tanh(v_logits[:, i, :]) + 1.0) / 2), 0.0, 1.0)
            o = torch.tanh(o_logits[:, i, :])

            h = torch.where(h < threshold, 0.0, h)
            h = torch.bernoulli(h)

            if self.kick_is_muted:
                h[:, 0::9] = 0
            if self.snare_is_muted:
                h[:, 1::9] = 0
            if self.hihat_is_muted:
                h[:, 2::9] = 0
                h[:, 3::9] = 0
            if self.tom_is_muted:
                h[:, 4::9] = 0
                h[:, 5::9] = 0
                h[:, 6::9] = 0
            if self.crash_is_muted:
                h[:, 7::9] = 0
            if self.ride_is_muted:
                h[:, 8::9] = 0

            self.generations[:, i, :] = torch.cat((h, v, o), dim=-1)
            self.num_generated_steps_available = max(self.num_generated_steps_available, i + 1)

            if self.decoder_input_has_velocity:
                self.shifted_tgt[:, i + 1, :] = torch.cat((h, v, o), dim=-1)
            else:
                self.shifted_tgt[:, i + 1, :] = torch.cat((h, torch.zeros_like(v) * 0, o), dim=-1)

        return self.generations[:, start_:start_ + n_steps_, :]

    @torch.jit.export
    def predict_n_steps_starting_at(self, start_: int, n_steps_:int, threshold: float = 0.05):
        """
        doesn't always generate the exact number of steps requested - will generate as many as possible
        ALWAYS check the number of generated steps using .shape[1] of the returned tensor
        """

        n_segs_in_bar = 16 // self.n_steps_per_segment

        if self.num_segments_encoded_so_far < n_segs_in_bar:
            print("Not enough segments encoded to generate a pattern - wait until more input is encoded")
            return torch.zeros((1, 0, 3 * self.num_tgt_voices))

        else:
            max_possible_field = self.num_segments_encoded_so_far * self.n_steps_per_segment + self.predict_K_bars_ahead * 16
            n_steps_ = min(n_steps_, max_possible_field - start_)

            # ensure the steps prior to start are already generated
            if self.num_generated_steps_available < 16:
                print("Need to generate the first bar before generating the requested steps")
                self.generations[:, :16, :] = self.predict_first_bar(threshold=threshold)
                self.num_generated_steps_available = 16

            if self.num_generated_steps_available < start_:
                print("num_generated_steps_available:", self.num_generated_steps_available, "start_:", start_)
                print("Need to generate some prior steps before generating the requested steps")
                i0 = self.num_generated_steps_available
                i1 = start_ - i0
                self.generations[:, i0:i1, :] = self.predict_n_steps_without_safety_checks(i0, i1, threshold=threshold)

            # ready to infere safely
            return self.predict_n_steps_without_safety_checks(start_, n_steps_, threshold=threshold)


    @torch.jit.export
    def predict_next_K_bars_starting_at(self, start_:int, threshold: float=0.05, print_info: bool=False):
        return self.predict_n_steps_starting_at(start_-self.total_shifts, self.predict_K_bars_ahead * 16, threshold=threshold)


    # @torch.jit.export
    # def predict_next_K_bars(self, roll_to_start_at_bar_boundary: bool=True, threshold: float=0.05, print_info: bool=False):
    #
    #     # get max number of steps available for prediction
    #     end_point = self.num_segments_encoded_so_far * self.n_steps_per_segment + self.predict_K_bars_ahead * 16
    #
    #     step_iters = self.predict_K_bars_ahead * 16 if end_point <= self.max_tgt_steps else self.max_tgt_steps - self.num_segments_encoded_so_far * self.n_steps_per_segment
    #
    #     with torch.no_grad():
    #         for i_ in range(step_iters):
    #
    #             i = i_ + self.num_segments_encoded_so_far * self.n_steps_per_segment
    #
    #             if i >= self.primed_for_N_segments * self.n_steps_per_segment:
    #
    #                 h_logits, v_logits, o_logits = self.DrumDecoder.forward(
    #                     tgt=self.shifted_tgt[:, :i+1, :],
    #                     memory=self.performance_memory[:, :self.num_segments_encoded_so_far, :]
    #                 )
    #
    #                 h_logits = h_logits[:, i, :] / self.temperature
    #                 h = torch.sigmoid(h_logits)
    #
    #                 # bernoulli sampling
    #                 v = torch.clamp(((torch.tanh(v_logits[:, i, :]) + 1.0) / 2), 0.0, 1.0)
    #                 o = torch.tanh(o_logits[:, i, :])
    #
    #                 h = torch.where(h < threshold, 0.0, h)
    #                 h = torch.bernoulli(h)
    #
    #                 if self.kick_is_muted:
    #                     h[:, 0::9] = 0
    #                 if self.snare_is_muted:
    #                     h[:, 1::9] = 0
    #                 if self.hihat_is_muted:
    #                     h[:, 2::9] = 0
    #                     h[:, 3::9] = 0
    #                 if self.tom_is_muted:
    #                     h[:, 4::9] = 0
    #                     h[:, 5::9] = 0
    #                     h[:, 6::9] = 0
    #                 if self.crash_is_muted:
    #                     h[:, 7::9] = 0
    #                 if self.ride_is_muted:
    #                     h[:, 8::9] = 0
    #
    #                 self.generations[:, i, :] = torch.cat((h, v, o), dim=-1)
    #
    #                 if self.decoder_input_has_velocity:
    #                     self.shifted_tgt[:, i+1, :] = torch.cat((h, v, o), dim=-1)
    #                 else:
    #                     self.shifted_tgt[:, i+1, :] = torch.cat((h, torch.zeros_like(v) * 0, o), dim=-1)
    #
    #         start_i = self.num_segments_encoded_so_far*self.n_steps_per_segment
    #
    #         gen = self.generations[:, start_i:start_i+self.predict_K_bars_ahead * 16, :].clone()
    #
    #         if print_info:
    #             print("self.num_segments_encoded_so_far", self.num_segments_encoded_so_far, "start_i", start_i, "generations.shape", gen.shape)
    #
    #         if roll_to_start_at_bar_boundary and start_i % 16 != 0:
    #             return torch.roll(gen, shifts=-(16 - (start_i % 16)), dims=1)
    #
    #         return gen

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


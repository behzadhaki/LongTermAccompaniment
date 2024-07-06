#  Copyright (c) 2024. \n Created by Behzad Haki. behzadhaki88@gmail.com

import torch
import json
import os

from model import TransformerEncoder, InputGrooveRhythmLayer, PositionalEncoding, SingleFeatureOutputLayer


class GrooveRhythmEncoder(torch.nn.Module):
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

        super(GrooveRhythmEncoder, self).__init__()

        self.config = config['GrooveEncoder'] if 'GrooveEncoder' in config else config

        # Layers
        # ---------------------------------------------------
        self.InputLayerEncoder = InputGrooveRhythmLayer(
            embedding_size=self.config['n_src_voices']*3,
            d_model=self.config['d_model'],
            max_len=16,
            velocity_dropout=float(self.config['velocity_dropout']),
            offset_dropout=float(self.config['offset_dropout']),
            positional_encoding_dropout=float(self.config['dropout']),
            has_velocity=self.config['has_velocity'],
            has_offset=self.config['has_offset']
        )

        self.Encoder = TransformerEncoder(
            d_model=self.config['d_model'],
            nhead=self.config['nhead'],
            dim_feedforward=self.config['dim_feedforward'],
            num_encoder_layers=self.config['n_layers'],
            dropout=float(self.config['dropout']),
            max_len=self.config['n_bars'] * 16
        )

        self.InputLayerEncoder.init_weights(0.1)

    @torch.jit.export
    def forward(self, src: torch.Tensor):
        """
        Encodes the input sequence through the encoder and predicts the latent space

        returns the last time step of the memory tensor

        :param src: [N, 16, 3]
        :return: memory: [N, d_model]

        """
        x, hit, hvo_projection = self.InputLayerEncoder(hvo=src)
        return self.Encoder.encode_unmasked(x)[:, -1, :]

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
                max_len=self.config['max_n_bars'],    # maximum number of bars in a performance
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

            self.max_n_bars = self.config['max_n_bars']

            self.causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(self.config['max_n_bars'])


            # self.Encoder = TransformerEncoder(
            #     d_model=self.config['d_model'],
            #     nhead=self.config['nhead'],
            #     dim_feedforward=self.config['dim_feedforward'],
            #     num_encoder_layers=self.config['n_layers'],
            #     dropout=float(self.config['dropout']),
            #     auto_regressive=True,
            #     max_len=self.config['max_n_bars']
            # )


        def create_src_mask(self, n_bars):
            # masked items are the ones noted as True
            # https://iori-yamahata.net/2024/02/28/programming-1-eng/

            batch_size = n_bars.shape[0]
            mask = torch.zeros((batch_size, self.max_n_bars)).bool()
            for i in range(batch_size):
                mask[i, n_bars[i]:] = 1
            return mask

        @torch.jit.export
        def forward(self, src_rhythm_encodings: torch.Tensor, src_key_padding_and_memory_mask: torch.Tensor):
            src = self.PositionalEncoding(src_rhythm_encodings)
            if self.auto_regressive:
                if self.causal_mask.device != src.device:
                    self.causal_mask = self.causal_mask.to(src.device)

                perf_encodings = self.Encoder.forward(
                    src=src,
                    src_key_padding_mask=src_key_padding_and_memory_mask,
                    is_causal=self.auto_regressive,
                    mask=self.causal_mask)
            else:
                perf_encodings = self.Encoder(src, src_key_padding_mask=src_key_padding_and_memory_mask)

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


class DrumContinuator(torch.nn.Module):

        def __init__(self, config):
            super(DrumContinuator, self).__init__()

            self.config = config['DrumContinuator'] if 'DrumContinuator' in config else config

            self.InputLayerEncoder = InputGrooveRhythmLayer(
                embedding_size=self.config['n_tgt_voices']*3,
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

        @torch.jit.export
        def forward(self, tgt: torch.Tensor, memory: torch.Tensor):
            x, hit, hvo_projection = self.InputLayerEncoder(hvo=tgt)

            # replace True to -inf, False to 0

            if self.causal_mask.device != x.device:
                self.causal_mask = self.causal_mask.to(x.device)

            output = self.Decoder(
                tgt=hvo_projection,
                memory=memory,
                tgt_mask=self.causal_mask
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


class LongTermAccompaniment(torch.nn.Module):

    def __init__(self, config):
        super(LongTermAccompaniment, self).__init__()
        self.config = config
        self.max_n_bars = self.config['PerformanceEncoder']['max_n_bars']
        self.encoder_d_model = self.config['PerformanceEncoder']['d_model']

        # The following will be shared every n_bars * 16 steps
        self.GrooveRhythmEncoder = GrooveRhythmEncoder(config)

        self.PerformanceEncoder = PerformanceEncoder(config)
        self.Performance2GrooveDecoder = DrumContinuator(config)

        # inference only utils
        self.encoded_bars = torch.zeros((1, self.max_n_bars, self.encoder_d_model))
        self.performance_memory = torch.zeros((1, self.max_n_bars, self.encoder_d_model))
        self.num_bars_encoded_so_far = 0
        self.max_tgt_steps = self.config['DrumContinuator']['max_steps']
        self.num_tgt_voices = self.config['DrumContinuator']['n_tgt_voices']
        self.hits_upcoming_playback = torch.zeros((1, self.max_tgt_steps, 1))
        self.velocities_upcoming_playback = torch.zeros((1, self.max_tgt_steps, 1))
        self.offsets_upcoming_playback = torch.zeros((1, self.max_tgt_steps, 1))


    def forward(self, src: torch.Tensor, src_key_padding_and_memory_mask: torch.Tensor, tgt: torch.Tensor):

        # GrooveRhythmEncoder
        n_bars = int(src.shape[1] // 16)
        encoded_bars = torch.zeros((src.shape[0], n_bars, self.encoder_d_model), device=src.device)
        for i in range(n_bars):
            encoded_bars[:, i, :] = self.GrooveRhythmEncoder.forward(
                src=src[:, i * 16: (i + 1) * 16, :])

        # PerformanceEncoder
        perf_encodings = self.PerformanceEncoder.forward(
            src_rhythm_encodings=encoded_bars,
            src_key_padding_and_memory_mask=src_key_padding_and_memory_mask)

        # check if batch size is 1
        if tgt.shape[0] == 1:
            i = src.shape[1]
            for i, val in enumerate(src_key_padding_and_memory_mask[0]):
                if val:
                    break
            perf_encodings = perf_encodings[:, :i, :]

        # decode the output groove
        h_logits, v_logits, o_logits = self.Performance2GrooveDecoder.forward(
            tgt=tgt,
            memory=perf_encodings)

        return h_logits, v_logits, o_logits

    @torch.jit.ignore
    def sample(self, src: torch.Tensor, src_key_padding_and_memory_mask: torch.Tensor, tgt: torch.Tensor):


        h_logits, v_logits, o_logits = self.forward(
            src=src,
            src_key_padding_and_memory_mask=src_key_padding_and_memory_mask,
            tgt=tgt
        )

        h = torch.sigmoid(h_logits)
        v = torch.tanh(v_logits) + 0.5
        o = torch.tanh(o_logits)

        h = torch.where(h > 0.5, 1, 0)

        return h, v, o, torch.cat((h, v, o), dim=-1)

    @torch.jit.export
    def prime_with_drum_pattern(self, drum_hvo: torch.Tensor):
        """
        Primes the model with a drum pattern
        """
        if len(drum_hvo.shape) == 2:
            drum_hvo = drum_hvo.unsqueeze(0)

        if (drum_hvo.shape[1] % 16 != 0):
            print(f'Error: The input drum pattern must have a multiple of 16 steps. The input has {drum_hvo.shape[1]} steps')

        n_bars = int(drum_hvo.shape[1] // 16)
        empty_instr_hvo = torch.zeros((1, 16, 3), dtype=torch.float32)

        for i in range(n_bars):
            self.encode_performed_bar(empty_instr_hvo, drum_hvo[:, i*16: (i+1)*16, :])
            self.moved_to_next_bar()

    @torch.jit.export
    def encode_performed_bar(self, instr_hvo: torch.Tensor, drum_hvo: torch.Tensor):
        """
        Encodes a single bar of the performance
        """

        if len(instr_hvo.shape) == 2:
            instr_hvo = instr_hvo.unsqueeze(0)

        if len(drum_hvo.shape) == 2:
            drum_hvo = drum_hvo.unsqueeze(0)

        if (instr_hvo.shape[1] != 16) or (drum_hvo.shape[1] != 16):
            print(f'Error: The input instrumental groove and the drums must have 16 steps (i.e. 1 bar). The input has {instr_hvo.shape[1]} steps')

        in_ = torch.cat((instr_hvo, drum_hvo), dim=-1)

        if self.training:
            self.eval()

        with torch.no_grad():
            self.encoded_bars[:, self.num_bars_encoded_so_far, :] = self.GrooveRhythmEncoder.forward(
                src=in_[:, :16, :])

            mask = torch.zeros((1, self.max_n_bars))
            mask[0, self.num_bars_encoded_so_far:] = 1

            self.performance_memory = self.PerformanceEncoder.forward(
                src_rhythm_encodings=self.encoded_bars,
                src_key_padding_and_memory_mask=mask)

    @torch.jit.export
    def encode_varying_length_performance(self, instr_hvo: torch.Tensor, drum_hvo: torch.Tensor):
        n_bars = int(instr_hvo.shape[1] // 16) +1
        for i in range(n_bars):
            self.encode_performed_bar(instr_hvo[:, i*16: (i+1)*16, :], drum_hvo[:, i*16: (i+1)*16, :])
            self.moved_to_next_bar()

    @torch.jit.export
    def moved_to_next_bar(self):
        if self.num_bars_encoded_so_far < self.max_n_bars:
            self.num_bars_encoded_so_far += 1
        elif self.num_bars_encoded_so_far == self.max_n_bars:
            self.shift_by_n_bars(1)
            self.num_bars_encoded_so_far = self.max_n_bars - 1

    @torch.jit.export
    def get_next_2_bars(self):
        self.hits_upcoming_playback = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
        self.velocities_upcoming_playback = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
        self.offsets_upcoming_playback = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))

        for i in range(self.max_tgt_steps):
            h_logits, v_logits, o_logits = self.Performance2GrooveDecoder.forward(
                tgt=torch.cat((self.hits_upcoming_playback, self.velocities_upcoming_playback, self.offsets_upcoming_playback), dim=-1),
                memory=self.performance_memory[:, :(self.num_bars_encoded_so_far + 1), :])

            step_h = torch.sigmoid(h_logits[:, i, :])
            # set over the threshold to 1
            step_v = torch.tanh(v_logits[:, i, :]) + 0.5
            step_o = torch.tanh(o_logits[:, i, :])

            current_step_hits = torch.where(step_h > 0.5, 1, 0)
            current_step_velocities = step_v * current_step_hits
            current_step_offsets = step_o * current_step_hits

            self.hits_upcoming_playback[:, i, :] = current_step_hits
            self.velocities_upcoming_playback[:, i, :] = current_step_velocities
            self.offsets_upcoming_playback[:, i, :] = current_step_offsets

        return self.hits_upcoming_playback, self.velocities_upcoming_playback, self.offsets_upcoming_playback, torch.concat((self.hits_upcoming_playback, self.velocities_upcoming_playback, self.offsets_upcoming_playback), dim=-1)


    @torch.jit.export
    def reset(self):
        self.encoded_bars = torch.zeros((1, self.max_n_bars, self.encoder_d_model))
        self.num_bars_encoded_so_far = 0

    @torch.jit.export
    def shift_by_n_bars(self, n_bars: int):
        # shift left by n_bars and fill the rest with zeros
        self.encoded_bars[:, :self.max_n_bars - n_bars, :] = self.encoded_bars[:, n_bars:, :]
        self.encoded_bars[:, self.max_n_bars - n_bars:, :] = 0

    @torch.jit.export
    def encode_next_performed_groove_bar(self, src: torch.Tensor):
        if (len(src.shape) == 2):
            src = src.unsqueeze(0)

        if (src.shape[1] != 16):
            print(f'Error: The input groove must have 16 steps. The input has {src.shape[1]} steps')

        self.encoded_bars[:, self.num_bars_encoded_so_far, :] = self.GrooveRhythmEncoder.forward(
            src=src[:, :16, :])

        # create a mask for the performance encoder
        mask = torch.zeros((1, self.max_n_bars))
        mask[0, self.num_bars_encoded_so_far:] = 1

        self.performance_memory = self.PerformanceEncoder.forward(
            src_rhythm_encodings=self.encoded_bars,
            src_key_padding_and_memory_mask=mask)


    @torch.jit.export
    def get_upcoming_2bars(self, sampling_thresh: float = 0.5):

        self.hits_upcoming_playback = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
        self.velocities_upcoming_playback = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))
        self.offsets_upcoming_playback = torch.zeros((1, self.max_tgt_steps, self.num_tgt_voices))

        for i in range(self.max_tgt_steps):
            h_logits, v_logits, o_logits = self.Performance2GrooveDecoder.forward(
                tgt=torch.cat((self.hits_upcoming_playback, self.velocities_upcoming_playback, self.offsets_upcoming_playback), dim=-1),
                memory=self.performance_memory)

            step_h = torch.sigmoid(h_logits[:, i, :])
            # set over the threshold to 1
            step_v = torch.tanh(v_logits[:, i, :]) + 0.5
            step_o = torch.tanh(o_logits[:, i, :])

            current_step_hits = torch.where(step_h > sampling_thresh, 1, 0)
            current_step_velocities = step_v * current_step_hits
            current_step_offsets =  step_o * current_step_hits

            self.hits_upcoming_playback[:, i, :] = current_step_hits
            self.velocities_upcoming_playback[:, i, :] = current_step_velocities
            self.offsets_upcoming_playback[:, i, :] = current_step_offsets

        return self.hits_upcoming_playback, self.velocities_upcoming_playback, self.offsets_upcoming_playback, torch.concat((self.hits_upcoming_playback, self.velocities_upcoming_playback, self.offsets_upcoming_playback), dim=-1)

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
    config = {
        'GrooveEncoder': {
            'd_model': 512,
            'dim_feedforward': 2048,
            'n_layers': 6,
            'nhead': 8,
            'n_src_voices': 1,
            'n_bars': 1,  # number of bars in a performance
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
            'max_n_bars': 32,  # maximum number of bars in a performance
            'dropout': 0.1,
            'positional_encoding_dropout': 0.1
        },
        'DrumContinuator': {
            'd_model': 512,
            'dim_feedforward': 2048,
            'n_layers': 8,
            'nhead': 4,
            'n_tgt_voices': 9,
            'max_steps': 32 * 1,
            'dropout': 0.1,
            'velocity_dropout': 0.1,
            'offset_dropout': 0.1,
            'positional_encoding_dropout': 0.1,
        }
    }
    max_bars = 32

    # GrooveRhythmEncoder
    # model = GrooveRhythmEncoder(config)
    #
    # mem = model.forward(
    #     src=torch.rand(1, 16, 3)
    # )
    #
    # print(mem.shape)
    #
    # model.serialize(save_folder='./')

    # PerformanceEncoder

    # model = PerformanceEncoder(config)
    # print(model)
    # n_bars = 128
    # perf_encodings = model(torch.rand(1, n_bars, 512), torch.tensor([[12]]))
    #
    # model.serialize(save_folder='./')
    #
    # # DrumContinuator
    # model = DrumContinuator(config)
    # print(model)
    #
    # h_logits, v_logits, o_logits = model(
    #     input_hvo=torch.rand(1, 32, 3),
    #     memory=perf_encodings
    # )
    #
    # print(h_logits.shape, v_logits.shape, o_logits.shape)
    #
    # model.serialize(save_folder='./')

    # LongTermAccompaniment
    model = LongTermAccompaniment(config)

    # serialize
    # model.serialize(save_folder='./misc')


    # b_size = 1
    # def create_src_mask(n_bars, max_n_bars):
    #     # masked items are the ones noted as True
    #
    #     batch_size = n_bars.shape[0]
    #     mask = torch.zeros((batch_size, max_n_bars)).bool()
    #     for i in range(batch_size):
    #         mask[i, n_bars[i]:] = 1
    #     return mask
    #
    # mask = create_src_mask(torch.tensor([12]*b_size), config['PerformanceEncoder']['max_n_bars'])
    #
    # mask[0]
    #
    # # time inference
    # from time import time
    # model.eval()
    # start = time()
    #
    # if config['GrooveEncoder']['has_velocity'] and config['GrooveEncoder']['has_offset']:
    #     n_features = 3
    # elif config['GrooveEncoder']['has_velocity']:
    #     n_features = 2
    # elif config['GrooveEncoder']['has_offset']:
    #     n_features = 2
    # else:
    #     n_features = 1
    #
    # with torch.no_grad():
    #     start = time()
    #
    #     h_logits, v_log, o_log = model.forward(
    #         src=torch.rand(b_size, 16 * config['PerformanceEncoder']['max_n_bars'], n_features * config['GrooveEncoder']['n_src_voices']),
    #         src_key_padding_and_memory_mask=mask,
    #         tgt=torch.rand(b_size, config['DrumContinuator']['max_steps'],
    #                        3 * config['DrumContinuator']['n_tgt_voices']))
    #     print("Time taken for inference: LTA.forward()", time() - start)
    #
    #
    # with torch.no_grad():
    #     start1 = time()
    #
    #     # encode new performed groove
    #     model.encode_next_performed_groove_bar(torch.rand(1, 16, config['GrooveEncoder']['n_src_voices']*n_features))
    #     h, v, o, hvo = model.get_upcoming_2bars()           # to be used in plugin
    #     print("Time taken for inference: LTA.encode_next_performed_groove_bar()", time() - start1)
    #
    #     start2 = time()
    #     # sampling during training
    #     h, v, o, hvo = model.sample(                        # use during testing
    #         src=torch.rand(b_size, 16 * config['PerformanceEncoder']['max_n_bars'], n_features * config['GrooveEncoder']['n_src_voices']),
    #         src_key_padding_and_memory_mask=mask,
    #         tgt=torch.rand(b_size, config['DrumContinuator']['max_steps'], 3 * config['DrumContinuator']['n_tgt_voices'])
    #     )
    #
    #     print("Time taken for inference: LTA.sample()", time() - start2)
    #
    #     print("Both inference took: ", time() - start1)

    #
    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset
    from data import PairedLTADataset

    # load dataset as torch.utils.data.Dataset
    training_dataset = PairedLTADataset(
        input_inst_dataset_bz2_filepath="data/lmd/data_bass_groove_test.bz2",
        output_inst_dataset_bz2_filepath="data/lmd/data_drums_full_unsplit.bz2",
        shift_tgt_by_n_steps=1,
        input_bars=config['PerformanceEncoder']['max_n_bars'],
        hop_n_bars=2,
        input_has_velocity=True,
        input_has_offsets=True
    )

    # access a batch
    previous_input_bars, upcoming_input_2_bars, previous_stacked_input_output_bars, upcoming_stacked_input_output_2_bars, previous_output_bars, upcoming_output_2_bars, shifted_upcoming_output_2_bars = training_dataset[2:3]

    def create_src_mask(n_bars, max_n_bars):
        # masked items are the ones noted as True

        batch_size = n_bars.shape[0]
        mask = torch.zeros((batch_size, max_n_bars)).bool()
        for i in range(batch_size):
            mask[i, n_bars[i]:] = 1
        return mask

    mask = create_src_mask(torch.tensor([12]*1), config['PerformanceEncoder']['max_n_bars'])

    h_logits, v_log, o_log = model.forward(
        src=previous_input_bars,
        src_key_padding_and_memory_mask=mask,
        tgt=shifted_upcoming_output_2_bars)

    #     h_logits, v_log, o_log = model.forward(
    #         src=torch.rand(b_size, 16 * config['PerformanceEncoder']['max_n_bars'], n_features * config['GrooveEncoder']['n_src_voices']),
    #         src_key_padding_and_memory_mask=mask,
    #         tgt=torch.rand(b_size, config['DrumContinuator']['max_steps'],
    #                        3 * config['DrumContinuator']['n_tgt_voices']))
    #     print("Time taken for inference: LTA.forward()", time() - start)
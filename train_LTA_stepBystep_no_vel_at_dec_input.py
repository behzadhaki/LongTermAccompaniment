import os

import wandb

import torch
from model.LongTermAccompanimentBeatwise import LongTermAccompanimentBeatwise
from helpers import train_utils #, eval_utils_g2g
from data.src.dataLoaders import PairedLTADatasetV2
from torch.utils.data import DataLoader
from logging import getLogger, DEBUG
import yaml
import argparse

logger = getLogger("train_**.py")
logger.setLevel(DEBUG)

# logger.info("MAKE SURE YOU DO THIS")
# logger.warning("this is a warning!")

parser = argparse.ArgumentParser()

# wandb parameters
parser.add_argument(
    "--config",
    help="Yaml file for configuratio. If available, the rest of the arguments will be ignored", default=None)

# ----------------------- SegmentEncoder Model Parameters -----------------------
parser.add_argument("--max_n_bars", type=int, help="Max Number of bars in the input/ouput", default=32)

parser.add_argument("--se_d_model", type=int, help="SegmentEncoder's d_model", default=128)
parser.add_argument("--se_d_ff", type=int, help="SegmentEncoder's d_ff", default=512)
parser.add_argument("--se_n_layers", type=int, help="SegmentEncoder's n_layers", default=3)
parser.add_argument("--se_n_heads", type=int, help="SegmentEncoder's n_heads", default=4)
parser.add_argument("--se_n_src1_voices", type=int, help="SegmentEncoder's n_src1_voices", default=1)
parser.add_argument("--se_n_src2_voices", type=int, help="SegmentEncoder's n_src2_voices - n_src2 is usually the same as target", default=0)
parser.add_argument("--se_steps_per_segment", type=int, help="SegmentEncoder's steps_per_segment", default=4)
parser.add_argument("--se_has_velocity", type=bool, help="SegmentEncoder's has_velocity", default=True)
parser.add_argument("--se_has_offset", type=bool, help="SegmentEncoder's has_offset", default=True)

parser.add_argument("--se_dropout", type=float, help="Dropout of SegmentEncoder transformer layers", default=0.1)
parser.add_argument("--se_velocity_dropout", type=float, help="Dropout of velocity information at the input of SegmentEncoder", default=0.1)
parser.add_argument("--se_offset_dropout", type=float, help="Dropout of offset information at the input of SegmentEncoder", default=0.1)
parser.add_argument("--se_positional_dropout", type=float, help="Dropout of positional encoding at the input of SegmentEncoder", default=0.1)


# ----------------------- PerformanceEncoder Model Parameters -----------------------
parser.add_argument("--pe_d_model", type=int, help="PerformanceEncoder's d_model", default=128)
parser.add_argument("--pe_d_ff", type=int, help="PerformanceEncoder's d_ff", default=512)
parser.add_argument("--pe_n_layers", type=int, help="PerformanceEncoder's n_layers", default=3)
parser.add_argument("--pe_n_heads", type=int, help="PerformanceEncoder's n_heads", default=4)
parser.add_argument("--pe_dropout", type=float, help="Dropout of PerformanceEncoder transformer layers", default=0.1)
parser.add_argument("--pe_positional_encoding_dropout", type=float, help="Dropout of positional encoding at the input of PerformanceEncoder", default=0.1)

# ----------------------- DrumDecoder Model Parameters -----------------------
parser.add_argument("--dc_d_model", type=int, help="DrumDecoder's d_model", default=128)
parser.add_argument("--dc_d_ff", type=int, help="DrumDecoder's d_ff", default=512)
parser.add_argument("--dc_n_layers", type=int, help="DrumDecoder's n_layers", default=3)
parser.add_argument("--dc_n_heads", type=int, help="DrumDecoder's n_heads", default=4)
parser.add_argument("--dc_n_tgt_voices", type=int, help="DrumDecoder's n_tgt_voices", default=9)
parser.add_argument("--dc_dropout", type=float, help="Dropout of DrumDecoder transformer layers", default=0.1)
parser.add_argument("--dc_velocity_dropout", type=float, help="Dropout of velocity information at the input of DrumDecoder", default=0.1)
parser.add_argument("--dc_offset_dropout", type=float, help="Dropout of offset information at the input of DrumDecoder", default=0.1)
parser.add_argument("--dc_positional_encoding_dropout", type=float, help="Dropout of positional encoding at the input of DrumDecoder", default=0.1)


# ----------------------- Training Parameters -----------------------
parser.add_argument("--scale_h_loss", type=float, help="Scale for hit loss", default=1)
parser.add_argument("--scale_v_loss", type=float, help="Scale for velocity loss", default=1)
parser.add_argument("--scale_o_loss", type=float, help="Scale for offset loss", default=1)
parser.add_argument("--optimizer", type=str, help="Optimizer to use", default="adam")
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.0001)
parser.add_argument("--batch_size", type=int, help="Batch Size", default=32)
parser.add_argument("--epochs", type=int, help="Number of epochs", default=1000)
parser.add_argument("--device", type=str, help="Device to run the model on", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--teacher_forcing_ratio", type=float, help="Teacher forcing ratio", default=0.5)

# ----------------------- Data Parameters -----------------------
parser.add_argument("--input_inst_dataset_bz2_filepath_train", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--output_inst_dataset_bz2_filepath_train", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--input_inst_dataset_bz2_filepath_test", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--output_inst_dataset_bz2_filepath_test", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--shift_tgt_by_n_steps", type=int,
                    help="Number of steps to shift the target by", default=1)
parser.add_argument("--hop_n_bars", type=int,
                    help="Number of bars to hop, if sequence is longer than max_n_beats, this value is used to scroll through and split the sequence", default=2)
parser.add_argument("--push_all_data_to_cuda", type=bool,
                    help="Push all data to device", default=True)

# ----------------------- Evaluation Params -----------------------
# parser.add_argument("--calculate_hit_scores_on_train", type=bool,
#                     help="Evaluates the quality of the hit models on training set",
#                     default=True)
# parser.add_argument("--calculate_hit_scores_on_test", type=bool,
#                     help="Evaluates the quality of the hit models on test/evaluation set",
#                     default=True)
# parser.add_argument("--piano_roll_samples", type=bool, help="Generate audio samples", default=True)
# parser.add_argument("--piano_roll_frequency", type=int, help="Frequency of piano roll generatio", default=10)
# parser.add_argument("--hit_score_frequency", type=int, help="Frequency of hit score generatio", default=5)

# ----------------------- Misc Params -----------------------
parser.add_argument("--save_model", type=bool, help="Save model", default=True)
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/LTA")
parser.add_argument("--upload_to_wandb", type=bool, help="Upload to wandb", default=True)
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=5)

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

loaded_via_config_yaml = False
if args.config is not None:
    with open(args.config, "r") as f:
        yml_ = yaml.safe_load(f)
        loaded_via_config_yaml = True

max_n_bars = args.max_n_bars if not loaded_via_config_yaml else yml_['max_n_bars']
steps_per_segment = args.se_steps_per_segment if not loaded_via_config_yaml else yml_['se_steps_per_segment']

hparams = {
    'scale_h_loss': args.scale_h_loss if not loaded_via_config_yaml else yml_['scale_h_loss'],
    'scale_v_loss': args.scale_v_loss if not loaded_via_config_yaml else yml_['scale_v_loss'],
    'scale_o_loss': args.scale_o_loss if not loaded_via_config_yaml else yml_['scale_o_loss'],
    'optimizer': args.optimizer if not loaded_via_config_yaml else yml_['optimizer'],
    'lr': args.lr if not loaded_via_config_yaml else yml_['lr'],
    'batch_size': args.batch_size if not loaded_via_config_yaml else yml_['batch_size'],
    'epochs': args.epochs if not loaded_via_config_yaml else yml_['epochs'],
    'device': args.device if not loaded_via_config_yaml else yml_['device'],
    'max_n_bars': args.max_n_bars if not loaded_via_config_yaml else yml_['max_n_bars'],
    'teacher_forcing_ratio': args.teacher_forcing_ratio if not loaded_via_config_yaml else yml_['teacher_forcing_ratio'],

    'SegmentEncoder': {
        'd_model': args.se_d_model if not loaded_via_config_yaml else yml_['se_d_model'],
        'dim_feedforward': args.se_d_ff if not loaded_via_config_yaml else yml_['se_d_ff'],
        'n_layers': args.se_n_layers if not loaded_via_config_yaml else yml_['se_n_layers'],
        'nhead': args.se_n_heads if not loaded_via_config_yaml else yml_['se_n_heads'],
        'n_src1_voices': args.se_n_src1_voices if not loaded_via_config_yaml else yml_['se_n_src1_voices'],
        'n_src2_voices': args.se_n_src2_voices if not loaded_via_config_yaml else yml_['se_n_src2_voices'],
        'steps_per_segment': args.se_steps_per_segment if not loaded_via_config_yaml else yml_['se_steps_per_segment'],
        'has_velocity': args.se_has_velocity if not loaded_via_config_yaml else yml_['se_has_velocity'],
        'has_offset': args.se_has_offset if not loaded_via_config_yaml else yml_['se_has_offset'],
        'dropout': args.se_dropout if not loaded_via_config_yaml else yml_['se_dropout'],
        'velocity_dropout': args.se_velocity_dropout if not loaded_via_config_yaml else yml_['se_velocity_dropout'],
        'offset_dropout': args.se_offset_dropout if not loaded_via_config_yaml else yml_['se_offset_dropout'],
        'positional_encoding_dropout': args.se_positional_dropout if not loaded_via_config_yaml else yml_['se_positional_dropout']
    },

    'PerformanceEncoder': {
        'd_model': args.pe_d_model if not loaded_via_config_yaml else yml_['pe_d_model'],
        'dim_feedforward': args.pe_d_ff if not loaded_via_config_yaml else yml_['pe_d_ff'],
        'n_layers': args.pe_n_layers if not loaded_via_config_yaml else yml_['pe_n_layers'],
        'nhead': args.pe_n_heads if not loaded_via_config_yaml else yml_['pe_n_heads'],
        'max_n_segments': max_n_bars * 16 // steps_per_segment,
        'dropout': args.pe_dropout if not loaded_via_config_yaml else yml_['pe_dropout'],
        'positional_encoding_dropout': args.pe_positional_encoding_dropout if not loaded_via_config_yaml else yml_['pe_positional_dropout']
    },

    'DrumDecoder': {
        'd_model': args.dc_d_model if not loaded_via_config_yaml else yml_['dc_d_model'],
        'dim_feedforward': args.dc_d_ff if not loaded_via_config_yaml else yml_['dc_d_ff'],
        'n_layers': args.dc_n_layers if not loaded_via_config_yaml else yml_['dc_n_layers'],
        'nhead': args.dc_n_heads if not loaded_via_config_yaml else yml_['dc_n_heads'],
        'n_tgt_voices': args.dc_n_tgt_voices if not loaded_via_config_yaml else yml_['dc_n_tgt_voices'],
        'max_steps': max_n_bars * 16,
        'dropout': args.dc_dropout if not loaded_via_config_yaml else yml_['dc_dropout'],
        'velocity_dropout': args.dc_velocity_dropout if not loaded_via_config_yaml else yml_['dc_velocity_dropout'],
        'offset_dropout': args.dc_offset_dropout if not loaded_via_config_yaml else yml_['dc_offset_dropout'],
        'positional_encoding_dropout': args.dc_positional_encoding_dropout if not loaded_via_config_yaml else yml_['dc_positional_dropout']
    },

    'input_inst_dataset_bz2_filepath_train': args.input_inst_dataset_bz2_filepath_train if not loaded_via_config_yaml else yml_['input_inst_dataset_bz2_filepath_train'],
    'output_inst_dataset_bz2_filepath_train': args.output_inst_dataset_bz2_filepath_train if not loaded_via_config_yaml else yml_['output_inst_dataset_bz2_filepath_train'],
    'input_inst_dataset_bz2_filepath_test': args.input_inst_dataset_bz2_filepath_test if not loaded_via_config_yaml else yml_['input_inst_dataset_bz2_filepath_test'],
    'output_inst_dataset_bz2_filepath_test': args.output_inst_dataset_bz2_filepath_test if not loaded_via_config_yaml else yml_['output_inst_dataset_bz2_filepath_test'],
    'shift_tgt_by_n_steps': args.shift_tgt_by_n_steps if not loaded_via_config_yaml else yml_['shift_tgt_by_n_steps'],
    'hop_n_bars': args.hop_n_bars if not loaded_via_config_yaml else yml_['hop_n_bars'],
    'push_all_data_to_cuda': args.push_all_data_to_cuda if not loaded_via_config_yaml else yml_['push_all_data_to_cuda'],
}


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,  # either from config file or CLI specified hyperparameters
        project="LTA_SegmentWise",
        entity="behzadhaki",  # saves in the mmil_vae_cntd team account
        settings=wandb.Settings(code_dir="scripts_archived/train_LTA_stepBystep.py"),
    )

    if loaded_via_config_yaml:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_LTA_stepBystep.py")
        wandb.run.log_artifact(model_code)

    # Reset config to wandb.config (in case of sweeping with YAML necessary)
    # ----------------------------------------------------------------------------------------------------------
    config = wandb.config
    print(config)
    run_name = wandb_run.name
    run_id = wandb_run.id

    # Load Training and Testing Datasets and Wrap them in torch.utils.data.Dataloader
    # ----------------------------------------------------------------------------------------------------------
    # only 1% of the dataset is used if we are testing the script (is_testing==True)
    # load dataset as torch.utils.data.Dataset
    training_dataset = PairedLTADatasetV2(
        input_inst_dataset_bz2_filepath=config['input_inst_dataset_bz2_filepath_train'],
        output_inst_dataset_bz2_filepath=config['output_inst_dataset_bz2_filepath_train'],
        shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
        max_input_bars=config['max_n_bars'],
        hop_n_bars=config['hop_n_bars'],
        input_has_velocity=config['SegmentEncoder']['has_velocity'],
        input_has_offsets=config['SegmentEncoder']['has_offset'],
        push_all_data_to_cuda=config['push_all_data_to_cuda']
    )
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    # load dataset as torch.utils.data.Dataset
    test_dataset = PairedLTADatasetV2(
        input_inst_dataset_bz2_filepath=config['input_inst_dataset_bz2_filepath_test'],
        output_inst_dataset_bz2_filepath=config['output_inst_dataset_bz2_filepath_test'],
        shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
        max_input_bars=config['max_n_bars'],
        hop_n_bars=config['hop_n_bars'],
        input_has_velocity=config['SegmentEncoder']['has_velocity'],
        input_has_offsets=config['SegmentEncoder']['has_offset'],
        push_all_data_to_cuda=config['push_all_data_to_cuda']
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    model_cpu = LongTermAccompanimentBeatwise(config)

    model_on_device = model_cpu.to(config.device)
    wandb.watch(model_on_device, log="all", log_freq=1)

    # Instantiate the loss Criterion and Optimizer
    # ------------------------------------------------------------------------------------------------------------

    hit_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    velocity_loss_fn = torch.nn.MSELoss(reduction='none')
    offset_loss_fn = torch.nn.MSELoss(reduction='none')

    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_on_device.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.SGD(model_on_device.parameters(), lr=config.lr)

    # Iterate over epochs
    # ------------------------------------------------------------------------------------------------------------
    metrics = dict()
    step_ = 0

    def create_src_mask(n_bars, max_n_bars):
        # masked items are the ones noted as True

        batch_size = n_bars.shape[0]
        mask = torch.zeros((batch_size, max_n_bars), dtype=torch.bool)
        for i in range(batch_size):
            mask[i, n_bars[i]:] = 1
        return mask

    # Batch Data IO Extractor
    def batch_data_extractor(data_, device=config.device):

        bass_solo = data_[0].to(device) if data_[0].device.type != device else data_[0]
        bass_solo[:, :, 1] = 0.0          # set vels to 0

        drums = data_[1].to(device) if data_[1].device.type != device else data_[1]

        stacked_bass_drums = None #data_[2].to(device) if data_[2].device.type != device else data_[2]

        shifted_drums = data_[3].to(device) if data_[3].device.type != device else data_[3]
        shifted_drums[:, :, 9:18] = 0.0         # set vels to 0

        return bass_solo, drums, stacked_bass_drums, shifted_drums


    def forward_using_batch_data(batch_data, model_=model_on_device, device=config.device):
        model_.train()

        bass_solo, drums, stacked_bass_drums, shifted_drums = batch_data_extractor(
            data_=batch_data,
            device='cpu'
        )

        # assert torch.all(shifted_drums[:, 1:, :] == drums[:, :-1, :])

        enc_src = bass_solo.to(device) if bass_solo.device.type != device else bass_solo

        if config['teacher_forcing_ratio'] >= 0.95:
            h_logits, v_log, o_log = model_.forward(
                src=enc_src,
                shifted_tgt=shifted_drums.to(device) if shifted_drums.device.type != device else shifted_drums)
            return h_logits, v_log, o_log, drums.to(device)
        # else:
        #     shifted_predicted_tgt = torch.zeros((drums.shape[0], drums.shape[1] + 16, drums.shape[2])).to(device)

        #     n_4_bars = shifted_predicted_tgt.shape[1] // (16)
        #
        #     for i in range(n_4_bars):
        #         h_logits, v_log, o_log = model_.forward(
        #             src=enc_src[:, :16*(i+1), :],
        #             shifted_tgt=shifted_predicted_tgt[:, :16*(i+1), :]
        #         )
        #         if torch.rand(1).item() > config['teacher_forcing_ratio']:
        #             h = torch.sigmoid(h_logits[:, -16:, :])
        #             # bernoulli sampling
        #             v = torch.clamp((torch.tanh(v_log[:, -16:, :]) + 1.0) / 2.0, 0.0, 1.0)
        #             o = torch.tanh(o_log[:, -16:, :])
        #             shifted_predicted_tgt[:, i*16:(i+1)*16, :] = torch.cat((h, v, o), dim=-1)
        #             del h, v, o
        #         else:
        #             shifted_predicted_tgt[:, i*16:(i+1)*16, :] = shifted_drums[:, i*16:(i+1)*16, :].to(device)

        # return h_logits, v_log, o_log, drums.to(device)


    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        model_on_device.train()

        logger.info("***************************Training...")

        train_log_metrics, step_ = train_utils.train_loop(
            train_dataloader=train_dataloader,
            forward_method=forward_using_batch_data,
            optimizer=optimizer,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            starting_step=step_,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss
        )

        wandb.log(train_log_metrics, commit=False)

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------------------------------------------
        # After each epoch, evaluate the model on the test set
        #     - To ensure not overloading the GPU, we evaluate the model on the test set also in batche
        #           rather than all at once
        # ---------------------------------------------------------------------------------------------------
        model_on_device.eval()  # DON'T FORGET TO SET THE MODEL TO EVAL MODE (check torch no grad)

        logger.info("***************************Testing...")

        test_log_metrics = train_utils.test_loop(
            test_dataloader=test_dataloader,
            forward_method=forward_using_batch_data,
            hit_loss_fn=hit_loss_fn,
            velocity_loss_fn=velocity_loss_fn,
            offset_loss_fn=offset_loss_fn,
            scale_h_loss=config.scale_h_loss,
            scale_v_loss=config.scale_v_loss,
            scale_o_loss=config.scale_o_loss
        )

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

        wandb.log(test_log_metrics, commit=False)
        logger.info(
            f"Epoch {epoch} Finished with total train loss of {train_log_metrics['Loss_Criteria/loss_recon_train']} "
            f"and test loss of {test_log_metrics['Loss_Criteria/loss_recon_test']}")

        wandb.log({"epoch": epoch}, step=epoch)

        # Save the model if needed
        # ---------------------------------------------------------------------------------------------------
        if args.save_model:
            if epoch % args.save_model_frequency == 0:
                if epoch < 10:
                    ep_ = f"00{epoch}"
                elif epoch < 100:
                    ep_ = f"0{epoch}"
                else:
                    ep_ = epoch
                model_artifact = wandb.Artifact(f'model_epoch_{ep_}', type='model')
                model_path = f"{args.save_model_dir}/{run_name}_{run_id}/{ep_}.pth"
                model_on_device.save(model_path)
                model_artifact.add_file(model_path)
                wandb_run.log_artifact(model_artifact)
                logger.info(f"Model saved to {model_path}")

        # empty gpu cache if cuda
        if config.device == 'cuda':
            torch.cuda.empty_cache()

    wandb.finish()

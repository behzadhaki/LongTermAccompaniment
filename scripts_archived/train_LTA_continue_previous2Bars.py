import os

import wandb

import torch
from model.LongTermAccompaniment import LongTermAccompanimentHierarchical
from helpers import train_utils #, eval_utils_g2g
from data.src.dataLoaders import PairedLTADataset
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

# ----------------------- GrooveEncoder Model Parameters -----------------------
parser.add_argument("--ge_d_model", type=int, help="GrooveEncoder's d_model", default=128)
parser.add_argument("--ge_d_ff", type=int, help="GrooveEncoder's d_ff", default=512)
parser.add_argument("--ge_n_layers", type=int, help="GrooveEncoder's n_layers", default=3)
parser.add_argument("--ge_n_heads", type=int, help="GrooveEncoder's n_heads", default=4)
parser.add_argument("--ge_n_src_voices", type=int, help="GrooveEncoder's n_src_voices", default=1)
parser.add_argument("--ge_nbars_per_segment", type=int, help="GrooveEncoder's nbars_per_segment: specifies how many bars are encoded using the groove encoder model", default=1)
parser.add_argument("--ge_has_velocity", type=bool, help="GrooveEncoder's has_velocity", default=True)
parser.add_argument("--ge_has_offset", type=bool, help="GrooveEncoder's has_offset", default=True)

parser.add_argument("--ge_dropout", type=float, help="Dropout of GrooveEncoder transformer layers", default=0.1)
parser.add_argument("--ge_velocity_dropout", type=float, help="Dropout of velocity information at the input of GrooveEncoder", default=0.1)
parser.add_argument("--ge_offset_dropout", type=float, help="Dropout of offset information at the input of GrooveEncoder", default=0.1)
parser.add_argument("--ge_positional_dropout", type=float, help="Dropout of positional encoding at the input of GrooveEncoder", default=0.1)


# ----------------------- PerformanceEncoder Model Parameters -----------------------
parser.add_argument("--pe_d_model", type=int, help="PerformanceEncoder's d_model", default=128)
parser.add_argument("--pe_d_ff", type=int, help="PerformanceEncoder's d_ff", default=512)
parser.add_argument("--pe_n_layers", type=int, help="PerformanceEncoder's n_layers", default=3)
parser.add_argument("--pe_n_heads", type=int, help="PerformanceEncoder's n_heads", default=4)
parser.add_argument("--pe_max_n_bars", type=int, help="PerformanceEncoder's max number of bars used for encoding", default=32)

parser.add_argument("--pe_dropout", type=float, help="Dropout of PerformanceEncoder transformer layers", default=0.1)
parser.add_argument("--pe_positional_encoding_dropout", type=float, help="Dropout of positional encoding at the input of PerformanceEncoder", default=0.1)

# ----------------------- DrumDecoder Model Parameters -----------------------
parser.add_argument("--dc_d_model", type=int, help="DrumDecoder's d_model", default=128)
parser.add_argument("--dc_d_ff", type=int, help="DrumDecoder's d_ff", default=512)
parser.add_argument("--dc_n_layers", type=int, help="DrumDecoder's n_layers", default=3)
parser.add_argument("--dc_n_heads", type=int, help="DrumDecoder's n_heads", default=4)
parser.add_argument("--dc_n_tgt_voices", type=int, help="DrumDecoder's n_tgt_voices", default=9)
parser.add_argument("--dc_max_steps", type=int, help="DrumDecoder's max_steps", default=32)

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
parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
parser.add_argument("--device", type=str, help="Device to run the model on", default="cuda" if torch.cuda.is_available() else "cpu")

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
parser.add_argument("--create_subsequences", type=bool,
                    help="Create subsequences from the max len input data", default=False)
parser.add_argument("--subsequence_hop_n_bars", type=int,
                    help="Number of bars to hop for the subsequences", default=1)
parser.add_argument("--push_all_data_to_cuda", type=bool,
                    help="Push all data to device", default=False)

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
parser.add_argument("--upload_to_wandb", type=bool, help="Upload to wandb", default=False)
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=1)

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

loaded_via_config_yaml = False
if args.config is not None:
    with open(args.config, "r") as f:
        yml_ = yaml.safe_load(f)
        loaded_via_config_yaml = True


hparams = {
    'scale_h_loss': args.scale_h_loss if not loaded_via_config_yaml else yml_['scale_h_loss'],
    'scale_v_loss': args.scale_v_loss if not loaded_via_config_yaml else yml_['scale_v_loss'],
    'scale_o_loss': args.scale_o_loss if not loaded_via_config_yaml else yml_['scale_o_loss'],
    'optimizer': args.optimizer if not loaded_via_config_yaml else yml_['optimizer'],
    'lr': args.lr if not loaded_via_config_yaml else yml_['lr'],
    'batch_size': args.batch_size if not loaded_via_config_yaml else yml_['batch_size'],
    'epochs': args.epochs if not loaded_via_config_yaml else yml_['epochs'],
    'device': args.device if not loaded_via_config_yaml else yml_['device'],

    'GrooveEncoder': {
        'd_model': args.ge_d_model if not loaded_via_config_yaml else yml_['ge_d_model'],
        'dim_feedforward': args.ge_d_ff if not loaded_via_config_yaml else yml_['ge_d_ff'],
        'n_layers': args.ge_n_layers if not loaded_via_config_yaml else yml_['ge_n_layers'],
        'nhead': args.ge_n_heads if not loaded_via_config_yaml else yml_['ge_n_heads'],
        'n_src_voices': args.ge_n_src_voices if not loaded_via_config_yaml else yml_['ge_n_src_voices'],
        'n_bars': args.ge_nbars_per_segment if not loaded_via_config_yaml else yml_['ge_nbars_per_segment'],
        'has_velocity': args.ge_has_velocity if not loaded_via_config_yaml else yml_['ge_has_velocity'],
        'has_offset': args.ge_has_offset if not loaded_via_config_yaml else yml_['ge_has_offset'],
        'dropout': args.ge_dropout if not loaded_via_config_yaml else yml_['ge_dropout'],
        'velocity_dropout': args.ge_velocity_dropout if not loaded_via_config_yaml else yml_['ge_velocity_dropout'],
        'offset_dropout': args.ge_offset_dropout if not loaded_via_config_yaml else yml_['ge_offset_dropout'],
        'positional_encoding_dropout': args.ge_positional_dropout if not loaded_via_config_yaml else yml_['ge_positional_dropout']
    },

    'PerformanceEncoder': {
        'd_model': args.pe_d_model if not loaded_via_config_yaml else yml_['pe_d_model'],
        'dim_feedforward': args.pe_d_ff if not loaded_via_config_yaml else yml_['pe_d_ff'],
        'n_layers': args.pe_n_layers if not loaded_via_config_yaml else yml_['pe_n_layers'],
        'nhead': args.pe_n_heads if not loaded_via_config_yaml else yml_['pe_n_heads'],
        'max_n_beats': args.pe_max_n_bars if not loaded_via_config_yaml else yml_['pe_max_n_bars'],
        'dropout': args.pe_dropout if not loaded_via_config_yaml else yml_['pe_dropout'],
        'positional_encoding_dropout': args.pe_positional_encoding_dropout if not loaded_via_config_yaml else yml_['pe_positional_dropout']
    },

    'DrumDecoder': {
        'd_model': args.dc_d_model if not loaded_via_config_yaml else yml_['dc_d_model'],
        'dim_feedforward': args.dc_d_ff if not loaded_via_config_yaml else yml_['dc_d_ff'],
        'n_layers': args.dc_n_layers if not loaded_via_config_yaml else yml_['dc_n_layers'],
        'nhead': args.dc_n_heads if not loaded_via_config_yaml else yml_['dc_n_heads'],
        'n_tgt_voices': args.dc_n_tgt_voices if not loaded_via_config_yaml else yml_['dc_n_tgt_voices'],
        'max_steps': args.dc_max_steps if not loaded_via_config_yaml else yml_['dc_max_steps'],
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
    'create_subsequences': args.create_subsequences if not loaded_via_config_yaml else yml_['create_subsequences'],
    'subsequence_hop_n_bars': args.subsequence_hop_n_bars if not loaded_via_config_yaml else yml_['subsequence_hop_n_bars'],
    'push_all_data_to_cuda': args.push_all_data_to_cuda if not loaded_via_config_yaml else yml_['push_all_data_to_cuda']
}


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,  # either from config file or CLI specified hyperparameters
        project="LTA_PreviousBarContinuator",
        entity="behzadhaki",  # saves in the mmil_vae_cntd team account
        settings=wandb.Settings(code_dir="train_LTA_continue_previous2Bars.py"),
    )

    if loaded_via_config_yaml:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("train_LTA_continue_previous2Bars.py")
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
    training_dataset = PairedLTADataset(
        input_inst_dataset_bz2_filepath=config['input_inst_dataset_bz2_filepath_train'],
        output_inst_dataset_bz2_filepath=config['output_inst_dataset_bz2_filepath_train'],
        shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
        max_input_bars=config['PerformanceEncoder']['max_n_beats'],
        continuation_bars=int(config['DrumDecoder']['max_steps'] // 16),
        hop_n_bars=config['hop_n_bars'],
        input_has_velocity=config['GrooveEncoder']['has_velocity'],
        input_has_offsets=config['GrooveEncoder']['has_offset'],
        push_all_data_to_cuda=config['push_all_data_to_cuda']
    )
    train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

    # load dataset as torch.utils.data.Dataset
    test_dataset = PairedLTADataset(
        input_inst_dataset_bz2_filepath=config['input_inst_dataset_bz2_filepath_test'],
        output_inst_dataset_bz2_filepath=config['output_inst_dataset_bz2_filepath_test'],
        shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
        max_input_bars=config['PerformanceEncoder']['max_n_beats'],
        continuation_bars=int(config['DrumDecoder']['max_steps'] // 16),
        hop_n_bars=config['hop_n_bars'],
        input_has_velocity=config['GrooveEncoder']['has_velocity'],
        input_has_offsets=config['GrooveEncoder']['has_offset'],
        push_all_data_to_cuda=config['push_all_data_to_cuda']
    )

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    model_cpu = LongTermAccompanimentHierarchical(config)

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
        mask = torch.zeros((batch_size, max_n_bars)).bool()
        for i in range(batch_size):
            mask[i, n_bars[i]:] = 1
        return mask

    # Batch Data IO Extractor
    def batch_data_extractor(data_, in_step_start, in_n_steps, out_step_start, out_n_steps, device=config.device):

        inst_1 = data_[0].to(device) if data_[0].device.type != device else data_[0]
        inst_2 = data_[1].to(device) if data_[1].device.type != device else data_[1]
        stacked_inst_12 = data_[2].to(device) if data_[2].device.type != device else data_[2]

        input_solo = inst_1[:, in_step_start:in_step_start + in_n_steps]
        input_stacked = stacked_inst_12[:, in_step_start:in_step_start + in_n_steps]
        next_output = inst_2[:, out_step_start:out_step_start + out_n_steps]
        previous_output = inst_2[:, out_step_start - out_n_steps:out_step_start]

        return input_solo, input_stacked, next_output, previous_output


    def predict_using_batch_data(batch_data, num_input_bars=None, model_=model_on_device, device=config.device):
        model_.eval()

        in_len = config['PerformanceEncoder']['max_n_beats'] * 16
        out_len = config['DrumDecoder']['max_steps']
        in_step_start = 0
        in_n_steps = in_len
        out_step_start = in_len
        out_n_steps = out_len

        input_solo, input_stacked, output, previous_output = batch_data_extractor(
            data_=batch_data,
            in_step_start=in_step_start,
            in_n_steps=in_n_steps,
            out_step_start=out_step_start,
            out_n_steps=out_n_steps,
            device=device
        )

        enc_src = input_stacked
        dec_src = previous_output

        if num_input_bars is None:
            num_input_bars = torch.ones((enc_src.shape[0], 1), dtype=torch.long).to(device) * config['PerformanceEncoder']['max_n_beats']

        with torch.no_grad():
            h, v, o, hvo = model_.sample(
                src=enc_src,
                src_key_padding_and_memory_mask=create_src_mask(num_input_bars, config['PerformanceEncoder']['max_n_beats']).to(device),
                tgt=dec_src
            )
        return hvo

    def forward_using_batch_data(batch_data, num_input_bars=None, model_=model_on_device, device=config.device):
        model_.train()

        in_len = config['PerformanceEncoder']['max_n_beats'] * 16
        out_len = config['DrumDecoder']['max_steps']
        in_step_start = 0
        in_n_steps = in_len
        out_step_start = in_len
        out_n_steps = out_len

        input_solo, input_stacked, output, previous_output = batch_data_extractor(
            data_=batch_data,
            in_step_start=in_step_start,
            in_n_steps=in_n_steps,
            out_step_start=out_step_start,
            out_n_steps=out_n_steps,
            device=device
        )

        enc_src = input_stacked
        dec_src = previous_output
        dec_tgt = output

        if num_input_bars is None:
            num_input_bars = torch.ones((enc_src.shape[0], 1), dtype=torch.long).to(device) * config['PerformanceEncoder']['max_n_beats']

        mask = create_src_mask(num_input_bars, config['PerformanceEncoder']['max_n_beats']).to(device)

        h_logits, v_log, o_log = model_.forward(
            src=enc_src,
            src_key_padding_and_memory_mask=mask,
            shifted_tgt=dec_src) # passing the previous 2 bars of drums as dec input and trying to predict the upcoming 2 bars

        return h_logits, v_log, o_log, dec_tgt

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

        # Generate PianoRolls and UMAP Plots  and KL/OA PLots if Needed
        # ---------------------------------------------------------------------------------------------------
        # if args.piano_roll_samples:
        #     if epoch % args.piano_roll_frequency == 0:
        #         logger.info("________Generating PianoRolls...")
        #         media, previous_evaluator_for_piano_rolls = eval_utils_g2g.get_pianoroll_for_wandb(
        #             model=model_on_device,
        #             predict_using_batch_data=predict_using_batch_data,
        #             dataset_bz2_filepath=config.bz2dataset_path,
        #             subset_name='test',
        #             down_sampled_ratio=0.1,
        #             cached_folder="cached/GrooveEvaluator/templates",
        #             divide_by_genre=True,
        #             previous_evaluator=previous_evaluator_for_piano_rolls,
        #             need_piano_roll=True,
        #             need_kl_plot=False,
        #             need_audio=False
        #         )
        #         wandb.log(media, commit=False)
        #
        #         # umap
        #         logger.info("________Generating UMAP...")
        #         media, previous_loaded_dataset_for_umap_test = eval_utils_g2g.generate_umap_for_wandb(
        #             predict_using_batch_data=predict_using_batch_data,
        #             dataset_bz2_filepath=config.bz2dataset_path,
        #             subset_name='test',
        #             previous_loaded_dataset=previous_loaded_dataset_for_umap_test,
        #             down_sampled_ratio=None,
        #         )
        #         if media is not None:
        #             wandb.log(media, commit=False)
        #
        #         media, previous_loaded_dataset_for_umap_train = eval_utils_g2g.generate_umap_for_wandb(
        #             predict_using_batch_data=predict_using_batch_data,
        #             dataset_bz2_filepath=config.bz2dataset_path,
        #             subset_name='train',
        #             previous_loaded_dataset=previous_loaded_dataset_for_umap_train,
        #             down_sampled_ratio=None,
        #         )
        #         if media is not None:
        #             wandb.log(media, commit=False)
        #
        # # Get Hit Scores for the entire train and the entire test set
        # # ---------------------------------------------------------------------------------------------------
        # if args.calculate_hit_scores_on_train:
        #     if epoch % args.hit_score_frequency == 0:
        #         logger.info("________Calculating Hit Scores on Train Set...")
        #         train_set_hit_scores, previous_evaluator_for_hit_scores_train = eval_utils_g2g.get_hit_scores(
        #             predict_using_batch_data=predict_using_batch_data,
        #             dataset_bz2_filepath=config.bz2dataset_path,
        #             subset_name='train',
        #             down_sampled_ratio=None,
        #             cached_folder="cached/GrooveEvaluator/templates",
        #             previous_evaluator=previous_evaluator_for_hit_scores_train,
        #             divide_by_genre=False
        #         )
        #         wandb.log(train_set_hit_scores, commit=False)
        #
        # if args.calculate_hit_scores_on_test:
        #     logger.info("________Calculating Hit Scores on Test Set...")
        #     test_set_hit_scores, previous_evaluator_for_hit_scores_test = eval_utils_g2g.get_hit_scores(
        #         predict_using_batch_data=predict_using_batch_data,
        #         dataset_bz2_filepath=config.bz2dataset_path,
        #         subset_name=args.evaluate_on_subset,
        #         down_sampled_ratio=None,
        #         cached_folder="cached/GrooveEvaluator/templates",
        #         previous_evaluator=previous_evaluator_for_hit_scores_test,
        #         divide_by_genre=False,
        #
        #     )
        #     wandb.log(test_set_hit_scores, commit=False)

        # Commit the metrics to wandb
        # ---------------------------------------------------------------------------------------------------
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

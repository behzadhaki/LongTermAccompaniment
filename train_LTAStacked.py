import os

import wandb

import torch
from model.LTA_Stacked import LTA_Stacked
from model.LTA_Stacked_MixedCausality import LTA_Stacked_MixedCausality
from helpers import train_utils #, eval_utils_g2g
from data.src.dataLoaders import StackedLTADatasetV2
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

parser.add_argument("--max_n_bars", type=int, help="Max Number of bars in the input/ouput", default=32)
parser.add_argument("--predict_K_bars_ahead", type=int, help="Predict K bars ahead", default=1)

# ----------------------- StepEncoder Model Parameters -----------------------

parser.add_argument("--d_model", type=int, help=" d_model", default=128)
parser.add_argument("--d_ff", type=int, help=" d_ff", default=512)
parser.add_argument("--n_layers", type=int, help=" n_layers", default=3)
parser.add_argument("--n_heads", type=int, help=" n_heads", default=4)
parser.add_argument("--n_src1_voices", type=int, help=" n_src1_voices", default=1)
parser.add_argument("--n_src2_voices", type=int, help=" n_src2_voices - n_src2 is usually the same as target", default=0)
parser.add_argument("--input_has_velocity", type=bool, help=" input_has_velocity", default=True)
parser.add_argument("--has_offset", type=bool, help=" has_offset", default=True)
parser.add_argument("--positional_encoding_dropout", type=float, help="Dropout of positional encoding at the input of PerformanceEncoder", default=0.1)
parser.add_argument("--dropout", type=float, help="Dropout of StepEncoder transformer layers", default=0.1)
parser.add_argument("--mixed_causality", type=bool, help="Mixed Causality", default=False)

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
parser.add_argument("--train_datasets", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--paired_drum_dataset_bz2_filepath_train", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--test_datasets", type=str,
                    help="Path to the dataset file in bz2 format")
parser.add_argument("--paired_drum_dataset_bz2_filepath_test", type=str,
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
parser.add_argument("--save_model_dir", type=str, help="Path to save the model", default="misc/LTA_Stacked")
parser.add_argument("--upload_to_wandb", type=bool, help="Upload to wandb", default=True)
parser.add_argument("--save_model_frequency", type=int, help="Save model every n epochs", default=5)
parser.add_argument("--run_name", type=str, help="Run name", default="")

args, unknown = parser.parse_known_args()
if unknown:
    logger.warning(f"Unknown arguments: {unknown}")

loaded_via_config_yaml = False
if args.config is not None:
    with open(args.config, "r") as f:
        yml_ = yaml.safe_load(f)
        loaded_via_config_yaml = True

max_n_bars = args.max_n_bars if not loaded_via_config_yaml else yml_['max_n_bars']
max_n_steps = max_n_bars * 16

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
    'max_n_steps': max_n_steps,
    'teacher_forcing_ratio': args.teacher_forcing_ratio if not loaded_via_config_yaml else yml_['teacher_forcing_ratio'],
    'mixed_causality': args.mixed_causality if not loaded_via_config_yaml else yml_['mixed_causality'],

    'd_model': args.d_model if not loaded_via_config_yaml else yml_['d_model'],
    'd_ff': args.d_ff if not loaded_via_config_yaml else yml_['d_ff'],
    'n_layers': args.n_layers if not loaded_via_config_yaml else yml_['n_layers'],
    'nhead': args.n_heads if not loaded_via_config_yaml else yml_['n_heads'],
    'n_src1_voices': args.n_src1_voices if not loaded_via_config_yaml else yml_['n_src1_voices'],
    'n_src2_voices': args.n_src2_voices if not loaded_via_config_yaml else yml_['n_src2_voices'],
    'dropout': args.dropout if not loaded_via_config_yaml else yml_['dropout'],
    'positional_encoding_dropout': args.positional_encoding_dropout if not loaded_via_config_yaml else yml_['positional_encoding_dropout'],
    'input_has_velocity': args.input_has_velocity if not loaded_via_config_yaml else yml_['input_has_velocity'],
    
    'train_datasets': args.train_datasets if not loaded_via_config_yaml else yml_['train_datasets'],
    'paired_drum_dataset_bz2_filepath_train': args.paired_drum_dataset_bz2_filepath_train if not loaded_via_config_yaml else yml_['paired_drum_dataset_bz2_filepath_train'],
    'test_datasets': args.test_datasets if not loaded_via_config_yaml else yml_['test_datasets'],
    'paired_drum_dataset_bz2_filepath_test': args.paired_drum_dataset_bz2_filepath_test if not loaded_via_config_yaml else yml_['paired_drum_dataset_bz2_filepath_test'],
    'shift_tgt_by_n_steps': args.shift_tgt_by_n_steps if not loaded_via_config_yaml else yml_['shift_tgt_by_n_steps'],
    'hop_n_bars': args.hop_n_bars if not loaded_via_config_yaml else yml_['hop_n_bars'],
    'push_all_data_to_cuda': args.push_all_data_to_cuda if not loaded_via_config_yaml else yml_['push_all_data_to_cuda'],
    'run_name': None if not loaded_via_config_yaml else yml_['run_name'] if 'run_name' in yml_ else None,
    'predict_K_bars_ahead': args.predict_K_bars_ahead if not loaded_via_config_yaml else yml_['predict_K_bars_ahead']
}

# check if the train_datasets and test_datasets are list, if not make them list
if not isinstance(hparams['train_datasets'], list):
    hparams['train_datasets'] = [hparams['train_datasets']]

if not isinstance(hparams['test_datasets'], list):
    hparams['test_datasets'] = [hparams['test_datasets']]


if __name__ == "__main__":

    # Initialize wandb
    # ----------------------------------------------------------------------------------------------------------
    wandb_run = wandb.init(
        config=hparams,  # either from config file or CLI specified hyperparameters
        project="LTA_Stacked_SegmentWise",
        entity="behzadhaki",  # saves in the mmil_vae_cntd team account
        settings=wandb.Settings(code_dir="train_LTAStacked.py"),
        name=hparams['run_name'] if hparams['run_name'] is not None else None
    )

    if loaded_via_config_yaml:
        model_code = wandb.Artifact("train_code_and_config", type="train_code_and_config")
        model_code.add_file(args.config)
        model_code.add_file("config_1barStacked.yaml")
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
    print("Loading Training Datasets: ", config['train_datasets'])

    if not config['mixed_causality']:   # LTA_Stacked
        training_datasets = []
        for dataset in config['train_datasets']:
            training_datasets.append(StackedLTADatasetV2(
                input_inst_dataset_bz2_filepath=dataset,
                output_inst_dataset_bz2_filepath=config['paired_drum_dataset_bz2_filepath_train'],
                shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
                max_input_bars=config['max_n_bars'],
                hop_n_bars=config['hop_n_bars'],
                push_all_data_to_cuda=config['push_all_data_to_cuda'],
                input_has_velocity=config['input_has_velocity'],
                start_with_one_bar_of_silent_drums=True,
            ))
        training_dataset = torch.utils.data.ConcatDataset(training_datasets)

        print(" |\n|\n--> Training Dataset Length: ", len(training_dataset))

        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

        # load dataset as torch.utils.data.Dataset
        testing_datasets = []
        for dataset in config['test_datasets']:
            testing_datasets.append(StackedLTADatasetV2(
                input_inst_dataset_bz2_filepath=dataset,
                output_inst_dataset_bz2_filepath=config['paired_drum_dataset_bz2_filepath_test'],
                shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
                max_input_bars=config['max_n_bars'],
                hop_n_bars=config['hop_n_bars'],
                push_all_data_to_cuda=config['push_all_data_to_cuda'],
                input_has_velocity=config['input_has_velocity'],
                start_with_one_bar_of_silent_drums=True,
            ))
        testing_dataset = torch.utils.data.ConcatDataset(testing_datasets)
    else:   # LTA_Stacked_MixedCausality
        training_datasets = []
        for dataset in config['train_datasets']:
            training_datasets.append(StackedLTADatasetV2(
                input_inst_dataset_bz2_filepath=dataset,
                output_inst_dataset_bz2_filepath=config['paired_drum_dataset_bz2_filepath_train'],
                shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
                max_input_bars=config['max_n_bars'],
                hop_n_bars=config['hop_n_bars'],
                push_all_data_to_cuda=config['push_all_data_to_cuda'],
                input_has_velocity=config['input_has_velocity'],
                start_with_one_bar_of_silent_drums=False,
            ))
        training_dataset = torch.utils.data.ConcatDataset(training_datasets)

        print(" |\n|\n--> Training Dataset Length: ", len(training_dataset))

        train_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)

        # load dataset as torch.utils.data.Dataset
        testing_datasets = []
        for dataset in config['test_datasets']:
            testing_datasets.append(StackedLTADatasetV2(
                input_inst_dataset_bz2_filepath=dataset,
                output_inst_dataset_bz2_filepath=config['paired_drum_dataset_bz2_filepath_test'],
                shift_tgt_by_n_steps=config['shift_tgt_by_n_steps'],
                max_input_bars=config['max_n_bars'],
                hop_n_bars=config['hop_n_bars'],
                push_all_data_to_cuda=config['push_all_data_to_cuda'],
                input_has_velocity=config['input_has_velocity'],
                start_with_one_bar_of_silent_drums=False,
            ))
        testing_dataset = torch.utils.data.ConcatDataset(testing_datasets)

    print(" |\n|\n--> Testing Dataset Length: ", len(testing_dataset))

    test_dataloader = DataLoader(testing_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize the model
    # ------------------------------------------------------------------------------------------------------------
    model_cpu = LTA_Stacked(config)

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

    printed_vel_info_already = False

    # Batch Data IO Extractor
    def batch_data_extractor(data_, device=config.device):
        global printed_vel_info_already
        stacked_target_shifted = data_[0].to(device)
        stacked_target = data_[1].to(device)


        if not printed_vel_info_already:
            print("##############################################")
            print("Stacked Target Shape: ", stacked_target.shape)
            print("Stacked Target Shifted Shape: ", stacked_target_shifted.shape)
            print("##############################################")
            printed_vel_info_already = True

        return stacked_target_shifted, stacked_target


    def forward_using_batch_data_teacher_force(batch_data, scope_end_step=None, model_=model_on_device, device=config.device):
        model_.train()

        teacher_forcing_ratio = config.teacher_forcing_ratio

        stacked_target_shifted, stacked_target = batch_data_extractor(
            data_=batch_data,
            device=device
        )

        if scope_end_step is not None:
            scope_end_step = min(scope_end_step, stacked_target_shifted.shape[1])
            stacked_target_shifted = stacked_target_shifted[:, :scope_end_step, :]
            stacked_target = stacked_target[:, :scope_end_step, :]

        h_logits, v_logits, o_logits = model_.forward_src_masked(
            shifted_tgt=stacked_target_shifted,
            teacher_forcing_ration=teacher_forcing_ratio)

        return h_logits, v_logits, o_logits, stacked_target.to(device)

    def forward_using_batch_data(batch_data, scope_end_step=None, model_=model_on_device, device=config.device):
        model_.train()

        stacked_target_shifted, stacked_target = batch_data_extractor(
            data_=batch_data,
            device=device
        )

        if scope_end_step is not None:
            scope_end_step = min(scope_end_step, stacked_target_shifted.shape[1])
            stacked_target_shifted = stacked_target_shifted[:, :scope_end_step, :]
            stacked_target = stacked_target[:, :scope_end_step, :]

        h_logits, v_logits, o_logits = model_.forward(shifted_tgt=stacked_target_shifted)

        return h_logits, v_logits, o_logits, stacked_target.to(device)


    for epoch in range(config.epochs):
        print(f"Epoch {epoch} of {config.epochs}, steps so far {step_}")

        # Run the training loop (trains per batch internally)
        # ------------------------------------------------------------------------------------------
        model_on_device.train()

        logger.info("***************************Training...")

        train_log_metrics, step_ = train_utils.train_loop(
            train_dataloader=train_dataloader,
            forward_method=forward_using_batch_data_teacher_force,
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

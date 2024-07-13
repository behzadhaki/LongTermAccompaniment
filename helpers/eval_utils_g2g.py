#  Copyright (c) 2022. \n Created by Behzad Haki. behzad.haki@upf.edu
import torch
import numpy as np
from eval.GrooveEvaluator import load_evaluator_template, load_g2g_evaluator_template
from eval.UMAP import UMapper
import tqdm
import time
from model.LongTermAccompaniment import LongTermAccompanimentHierarchical

from logging import getLogger
logger = getLogger("helpers.LongTermAccompanimentHierarchical.eval_utils")
logger.setLevel("DEBUG")
from data import PairedLTADataset


def get_pianoroll_for_wandb(
        predict_using_batch_data,
        dataset_bz2_filepath, subset_name,
        down_sampled_ratio,
        cached_folder="cached/Evaluators/templates/",
        divide_by_genre=True,
        previous_evaluator=None,
        bz2_dataset=None,
        **kwargs):
    """
    Prepare the media for logging in wandb. Can be easily used with an evaluator template
    (A template can be created using the code in eval/GrooveEvaluator/templates/main.py)
    :param predict_using_batch_data: The function to be used for prediction
    :param dataset_bz2_filepath: The path to the dataset setting json file
    :param subset_name: The name of the subset to be evaluated
    :param down_sampled_ratio: The ratio of the subset to be evaluated
    :param cached_folder: The folder to be used for caching the evaluator template
    :param divide_by_genre: Whether to divide the subset by genre or not
    :param previous_evaluator: The previous evaluator to be used for logging (this optimizes the loading/creating of the evaluator). In the second epoch, pass the returned evaluator from the first epoch.
    :param kwargs:                  additional arguments: need_hit_scores, need_velocity_distributions,
                                    need_offset_distributions, need_rhythmic_distances, need_heatmap
    :return:                        a ready to use dictionary to be logged using wandb.log()
    """

    start = time.time()

    if previous_evaluator is not None:
        evaluator = previous_evaluator
    else:
        # load the evaluator template (or create a new one if it does not exist)
        evaluator = load_g2g_evaluator_template(
            dataset_bz2_filepath=dataset_bz2_filepath,
            subset_name=subset_name,
            down_sampled_ratio=down_sampled_ratio,
            cached_folder=cached_folder,
            divide_by_genre=divide_by_genre
        )
    batch_data = evaluator.dataset[:]
    full_midi_filenames = [hvo_seq.metadata["full_midi_filename"] for hvo_seq in evaluator.dataset.hvo_sequences]

    try:
        hvos_array, _ = predict_using_batch_data(batch_data=batch_data)
    except:
        hvos_array = predict_using_batch_data(batch_data=batch_data)

    evaluator.add_unsorted_predictions(hvos_array.detach().cpu().numpy(), full_midi_filenames)

    # Get the media from the evaluator
    # -------------------------------
    media = evaluator.get_logging_media(
        prepare_for_wandb=True,
        need_hit_scores=kwargs["need_hit_scores"] if "need_hit_scores" in kwargs.keys() else False,
        need_velocity_distributions=kwargs["need_velocity_distributions"] if "need_velocity_distributions" in kwargs.keys() else False,
        need_offset_distributions=kwargs["need_offset_distributions"] if "need_offset_distributions" in kwargs.keys() else False,
        need_rhythmic_distances=kwargs["need_rhythmic_distances"] if "need_rhythmic_distances" in kwargs.keys() else False,
        need_heatmap=kwargs["need_heatmap"] if "need_heatmap" in kwargs.keys() else False,
        need_global_features=kwargs["need_global_features"] if "need_global_features" in kwargs.keys() else False,
        need_piano_roll=kwargs["need_piano_roll"] if "need_piano_roll" in kwargs.keys() else False,
        need_audio=kwargs["need_audio"] if "need_audio" in kwargs.keys() else False,
        need_kl_oa=kwargs["need_kl_oa"] if "need_kl_oa" in kwargs.keys() else False)

    end = time.time()
    logger.info(f"PianoRoll Generation for {subset_name} took {end - start} seconds")

    return media, evaluator


def get_hit_scores(
        predict_using_batch_data, dataset_bz2_filepath, subset_name,
        down_sampled_ratio,
        cached_folder="cached/Evaluators/templates/",
        previous_evaluator=None,
        divide_by_genre=True):

    # logger.info("Generating the hit scores for subset: {}".format(subset_name))
    # and model is correct type

    start = time.time()

    if previous_evaluator is not None:
        evaluator = previous_evaluator
    else:
        # load the evaluator template (or create a new one if it does not exist)
        evaluator = load_g2g_evaluator_template(
            dataset_bz2_filepath=dataset_bz2_filepath,
            subset_name=subset_name,
            down_sampled_ratio=down_sampled_ratio,
            cached_folder=cached_folder,
            divide_by_genre=divide_by_genre
        )

    print(f"evaluator = load_evaluator_template("
          f"dataset_bz2_filepath={dataset_bz2_filepath},"
          f"subset_name={subset_name},"
          f"down_sampled_ratio={down_sampled_ratio},"
          f"cached_folder={cached_folder},"
          f"divide_by_genre={divide_by_genre}")

    # (1) Get the targets, (2) tapify and pass to the model (3) add the predictions to the evaluator
    # ------------------------------------------------------------------------------------------
    dataloader = torch.utils.data.DataLoader(
        evaluator.dataset,
        batch_size=128,
        shuffle=False,
    )

    predictions = []


    try:
        for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Generating Hit Scores - {subset_name}"):
            hvos_array, latent_z = predict_using_batch_data(batch_data=batch_data)
            predictions.append(hvos_array.detach().cpu().numpy())
    except:
        for batch_ix, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Generating Hit Scores - {subset_name}"):
            hvos_array = predict_using_batch_data(batch_data=batch_data)
            predictions.append(hvos_array.detach().cpu().numpy())

    full_midi_filenames = [hvo_seq.metadata["full_midi_filename"] for hvo_seq in evaluator.dataset.hvo_sequences]


    evaluator.add_unsorted_predictions(np.concatenate(predictions), full_midi_filenames)

    hit_dict = evaluator.get_statistics_of_pos_neg_hit_scores()

    score_dict = {f"Hit_Scores/{key}_mean_{subset_name}".replace(" ", "_").replace("-", "_"): float(value['mean']) for key, value
                  in sorted(hit_dict.items())}

    score_dict.update({f"Hit_Scores/{key}_std_{subset_name}".replace(" ", "_").replace("-", "_"): float(value['std']) for key, value
                  in sorted(hit_dict.items())})

    end = time.time()
    logger.info(f"Hit Scores Generation for {subset_name} took {end - start} seconds")
    return score_dict, evaluator

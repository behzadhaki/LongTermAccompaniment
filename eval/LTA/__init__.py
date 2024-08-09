import torch
from sklearn.metrics import confusion_matrix
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# model loader
def load_classifier_model(model_path, model_class, params_dict=None, is_evaluating=True, device=None):
    """ Load a GenreGlobalDensityWithVoiceMutesVAE model from a given path

    Args:
        model_path (str): path to the model
        params_dict (None, dict, or json path): dictionary containing the parameters of the model
        is_evaluating (bool): if True, the model is set to eval mode
        device (None or torch.device): device to load the model to (if cpu, the model is loaded to cpu)

    Returns:
        model (GenreDensityTempoVAE): the loaded model
    """

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


import wandb, os
from model import GenreClassifier
import shutil

def predict_genres_using_hvo_and_model(hvos, model):
    model.eval()
    with torch.no_grad():
        predicted_genres, predicted_probs_per_genre = model.predict(hvos)

    # sample 100 times from probs
    sampled_genres = []
    genres = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(1):
        for probs in predicted_probs_per_genre:
            sampled_genres.append(np.random.choice(genres, p=probs.cpu().numpy()))

    return np.array(sampled_genres), predicted_probs_per_genre

from bokeh.layouts import row
from bokeh.models import Panel, Tabs

def get_genre_accuracy_of_patterns(hvo1, hvo2, model_classifier, indices=None):

        # predict genres
        predicted_genres1, predicted_probs_per_genre1 = predict_genres_using_hvo_and_model(hvo1, model_classifier)
        predicted_genres2, predicted_probs_per_genre2 = predict_genres_using_hvo_and_model(hvo2, model_classifier)

        return (predicted_genres1==predicted_genres2)[0]


def get_subsets_hit_vel_stats(hvos):
    vels_at_hits = []
    offs_at_hits = []
    average_hits_per_sample = []
    n_dim_per_feature = hvos.shape[-1] // 3
    print(n_dim_per_feature)
    # index of non_zero hits (shape
    hits = hvos[:, :, :n_dim_per_feature].clone()
    vels = hvos[:, :, n_dim_per_feature:2 * n_dim_per_feature].clone()
    offs = hvos[:, :, 2 * n_dim_per_feature:].clone()

    vels_at_hits = vels[hits == 1].view(-1)
    offs_at_hits = offs[hits == 1].view(-1)

    average_hits_per_sample = hits.sum(dim=1).sum(dim=1).mean().item()
    std_hits_per_sample = hits.sum(dim=1).sum(dim=1).std().item()

    return f'{vels_at_hits.mean().item():.2f} ± {vels_at_hits.std().item():.2f}', f'{offs_at_hits.mean().item():.2f} ± {offs_at_hits.std().item():.2f}', f'{average_hits_per_sample:.2f} ± {std_hits_per_sample:.2f}'

from hvo_sequence.hvo_seq import HVO_Sequence
from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
def get_overlapping_ration(hvo_drum, hvo_groove):
    drum_hits = hvo_drum[:, :, :9].clone()
    groove_hits = hvo_groove[:, :, 1].clone()

    drum_hits_flattened = drum_hits.sum(dim=-1).clip(0, 1)

    overlapping_step_counts = (drum_hits_flattened * groove_hits).sum(-1) / drum_hits_flattened.sum(-1) * 100

    # xor
    complementary_ratio = (drum_hits_flattened * (1 - groove_hits)).sum(-1) / drum_hits_flattened.sum(-1) * 100
    # replace nans with 0
    overlapping_step_counts[torch.isnan(overlapping_step_counts)] = 0

    return f"{overlapping_step_counts.mean().item():.2f} ± {overlapping_step_counts.std().item():.2f} %"


def get_as_hvo_sequence(hvo, is_groove=False):
    if not is_groove:
        hvo_seq = HVO_Sequence(
            beat_division_factors=[4],
            drum_mapping=ROLAND_REDUCED_MAPPING
        )
    else:
        hvo_seq = HVO_Sequence(
            beat_division_factors=[4],
            drum_mapping={"groove": [x for x in range(127)]},
        )

    hvo_seq.add_tempo(0, 120)
    hvo_seq.add_time_signature(0, 4, 4)
    hvo_seq.hvo = hvo.cpu().numpy()
    return hvo_seq


def jaccard_similarity(hvo1, hvo2, has_batch_dim=True):
    if has_batch_dim:
        assert len(hvo1.shape) == 3 and len(hvo2.shape) == 3, "make sure you have batch dimension"
        n_voices = hvo1.shape[-1] // 3
        hvo_flat1 = torch.flatten(hvo1[:,:,:n_voices], start_dim=1)
        hvo_flat2 = torch.flatten(hvo2[:,:,:n_voices], start_dim=1)
        overlap_hits = hvo_flat1 * hvo_flat2
        overlap_sum = torch.sum(overlap_hits, dim=1)
        union_hits = hvo_flat1 + hvo_flat2 - overlap_hits
        union_sum = torch.sum(union_hits, dim=1)
        jaccard = overlap_sum / union_sum
        return jaccard
    else:
        assert len(hvo1.shape) == 2 and len(hvo2.shape) == 2, "make sure you dont have batch dimension"
        n_voices = hvo1.shape[-1] // 3
        hvo_flat1 = torch.flatten(hvo1[:,:n_voices])
        hvo_flat2 = torch.flatten(hvo2[:,:n_voices])
        overlap_hits = hvo_flat1 * hvo_flat2
        overlap_sum = torch.sum(overlap_hits)
        union_hits = hvo_flat1 + hvo_flat2 - overlap_hits
        union_sum = torch.sum(union_hits)
        jaccard = overlap_sum / union_sum
        return jaccard


def velocity_distribution(hvo, only_at_hits=False, average_per_sample_first=False):
    assert len(hvo.shape) == 3, "make sure you have batch dimension"
    n_voices = hvo.shape[-1] // 3
    if not only_at_hits:
        vel = hvo[:, :, n_voices:n_voices*2]
    else:
        hits = hvo[:, :, :n_voices]
        vel = hvo[:, :, n_voices:n_voices*2]
        vel = vel[hits > 0.5]

    if average_per_sample_first:
        vel = vel.mean(dim=-1).mean(dim=-1)
        print(vel.shape)

    vel_mean = torch.mean(vel).item()
    vel_std = torch.std(vel).item()

    return np.round(vel_mean, 2), np.round(vel_std, 2)

def velocity_error(hvo1, hvo2, only_at_hits=False):
    assert len(hvo1.shape) == 3 and len(hvo2.shape) == 3, "make sure you have batch dimension"

    n_voices = hvo2.shape[-1] // 3

    if not only_at_hits:
        vel1 = hvo1[:, :, n_voices:n_voices*2]
        vel2 = hvo2[:, :, n_voices:n_voices*2]
    else:
        hits1 = hvo1[:, :, :n_voices]
        hits2 = hvo2[:, :, :n_voices]
        vel1 = hvo1[:, :, n_voices:n_voices*2]
        vel2 = hvo2[:, :, n_voices:n_voices*2]
        vel1 = vel1[(hits1 + hits2) > 0.5]
        vel2 = vel2[(hits1 + hits2) > 0.5]

    vel_diff = torch.abs(vel1 - vel2)
    print(vel_diff.shape)
    vel_diff_mean = torch.mean(vel_diff).item()
    vel_diff_std = torch.std(vel_diff).item()

    return np.round(vel_diff_mean, 2), np.round(vel_diff_std, 2)

def offset_distribution(hvo, only_at_hits=False, average_per_sample_first=False):
    assert len(hvo.shape) == 3, "make sure you have batch dimension"
    n_voices = hvo.shape[-1] // 3
    if not only_at_hits:
        offset = hvo[:, :, (n_voices*2):]
    else:
        hits = hvo[:, :, :n_voices]
        offset = hvo[:, :, (n_voices*2):]
        offset = offset[hits > 0.5]

    if average_per_sample_first:
        offset = offset.mean(dim=-1).mean(dim=-1)

    offset_mean = torch.mean(offset).item()
    offset_std = torch.std(offset).item()

    return np.round(offset_mean, 2), np.round(offset_std, 2)

import tqdm

def extract_SSM_data(generated_drums_, gt_drums_, get_flatten=False, ignore_first_n_bars=2, shift_frames_by=1, load_n_bars=8, steps_per_segment=1, quantized_analysis=False,):
    # load generations
    ignore_first_n_bars_ = max(2, ignore_first_n_bars)
    end_step = (load_n_bars + ignore_first_n_bars_) * 16 // shift_frames_by
    generated_drums = generated_drums_.clone()[:, ignore_first_n_bars_ * 16:(load_n_bars + ignore_first_n_bars_) * 16, :9]
    gt_drums = gt_drums_.clone()[:, ignore_first_n_bars_ * 16:(load_n_bars + ignore_first_n_bars_) * 16, :9]

    if get_flatten:
        generated_drums = torch.clamp(generated_drums.sum(dim=-1), 0, 1).reshape(generated_drums.shape[0], -1, 1)
        gt_drums = torch.clamp(gt_drums.sum(dim=-1), 0, 1).reshape(gt_drums.shape[0], -1, 1)

    def jaccard(hvo_1, hvo_2):
        hvo_1 = hvo_1.reshape(hvo_1.shape[0], -1)
        hvo_2 = hvo_2.reshape(hvo_2.shape[0], -1)
        intersection = torch.sum(hvo_1 * hvo_2, dim=1)
        union = torch.sum(hvo_1 + hvo_2, dim=1)
        jac = intersection / (union - intersection)
        # replace nans with 0
        jac[torch.isnan(jac)] = 1
        return jac

    def cosine_similarity(hvo_1, hvo_2):
        hvo_1 = hvo_1.reshape(hvo_1.shape[0], -1)
        hvo_2 = hvo_2.reshape(hvo_2.shape[0], -1)
        return torch.sum(hvo_1 * hvo_2, dim=1) / (torch.norm(hvo_1, dim=1) * torch.norm(hvo_2, dim=1))

    def extract_segments(hvo_, n_bars_per_seg):
        n_bars = hvo_.shape[1] // 16
        n_segs = n_bars // n_bars_per_seg
        hvo_ = hvo_[:, :n_segs * n_bars_per_seg * 16, :]
        hvo_ = hvo_.reshape(hvo_.shape[0], n_segs, n_bars_per_seg * 16, hvo_.shape[2])
        return hvo_

    def compute_intra_jaccards(hvo_, n_bars_per_seg):
        hvo_segments = extract_segments(hvo_, n_bars_per_seg)
        N = hvo_segments.shape[1]
        intra_jaccards = torch.zeros(hvo_segments.shape[0], N * (N - 1) // 2)
        count = 0
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                intra_jaccards[:, count] = jaccard(hvo_segments[:, i, :, :], hvo_segments[:, j, :, :])
                count += 1
        return intra_jaccards

    def compute_inter_jaccards(hvo_1, hvo_2, n_bars_per_seg):
        hvo_1_segments = extract_segments(hvo_1, n_bars_per_seg)
        hvo_2_segments = extract_segments(hvo_2, n_bars_per_seg)
        N = hvo_1_segments.shape[1]
        inter_jaccards = torch.zeros(hvo_1_segments.shape[0], N)

        for i in range(N):
            inter_jaccards[:, i] = jaccard(hvo_1_segments[:, i, :, :], hvo_2_segments[:, i, :, :])

        return inter_jaccards

    def quantize_jaccards(jaccards, n_bins=3):
        return torch.floor(jaccards * n_bins) / n_bins

    n_segments = 0
    for i in tqdm.trange(0, generated_drums.shape[1] - steps_per_segment - shift_frames_by, shift_frames_by):
        for j in range(0, generated_drums.shape[1] - steps_per_segment - shift_frames_by, shift_frames_by):
            n_segments += 1
        break

    print(n_segments)
    SSM_Generated = torch.zeros(generated_drums.shape[0], n_segments, n_segments)
    SSM_GT = torch.zeros(generated_drums.shape[0], n_segments, n_segments)
    CSM = torch.zeros(generated_drums.shape[0], n_segments, n_segments)

    ssm_i = 0
    for i in tqdm.trange(0, generated_drums.shape[1] - steps_per_segment - shift_frames_by, shift_frames_by):
        ssm_j = 0
        for j in range(0, generated_drums.shape[1] - steps_per_segment - shift_frames_by, shift_frames_by):
            if quantized_analysis:
                start_1 = i
                end_1 = i + steps_per_segment
                start_2 = j
                end_2 = j + steps_per_segment
                SSM_Generated[:, ssm_i, ssm_j] = quantize_jaccards(
                    jaccard(generated_drums[:, start_1:end_1, :], generated_drums[:, start_2:end_2, :]))
                SSM_GT[:, ssm_i, ssm_j] = quantize_jaccards(
                    jaccard(gt_drums[:, start_1:end_1, :], gt_drums[:, start_2:end_2, :]))
            else:
                start1 = i
                end1 = i + steps_per_segment
                start2 = j
                end2 = j + steps_per_segment
                SSM_Generated[:, ssm_i, ssm_j] = jaccard(generated_drums[:, start1:end1, :],
                                                         generated_drums[:, start2:end2, :])
                SSM_GT[:, ssm_i, ssm_j] = jaccard(gt_drums[:, start1:end1, :], gt_drums[:, start2:end2, :])
            ssm_j += 1
        ssm_i += 1

    # cross similarity matrix (csm)
    ssm_i = 0
    for i in tqdm.trange(0, generated_drums.shape[1] - steps_per_segment - shift_frames_by, shift_frames_by):
        ssm_j = 0
        for j in range(0, generated_drums.shape[1] - steps_per_segment - shift_frames_by, shift_frames_by):
            if quantized_analysis:
                start_1 = i
                end_1 = i + steps_per_segment
                start_2 = j
                end_2 = j + steps_per_segment
                CSM[:, ssm_i, ssm_j] = quantize_jaccards(
                    jaccard(generated_drums[:, start_1:end_1, :], gt_drums[:, start_2:end_2, :]))
            else:
                start1 = i
                end1 = i + steps_per_segment
                start2 = j
                end2 = j + steps_per_segment
                CSM[:, ssm_i, ssm_j] = jaccard(generated_drums[:, start1:end1, :], gt_drums[:, start2:end2, :])
            ssm_j += 1
        ssm_i += 1

    return SSM_Generated, SSM_GT, CSM, generated_drums, gt_drums

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

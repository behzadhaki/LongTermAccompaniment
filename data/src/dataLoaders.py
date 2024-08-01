from data.src.utils import (get_data_directory_using_filters, get_drum_mapping_using_label,
                            load_original_gmd_dataset_pickle, extract_hvo_sequences_dict, pickle_hvo_dict)
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from math import ceil
import json
import os
import pickle
import bz2
import logging
import random
from data import *
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

logging.basicConfig(level=logging.DEBUG)
dataLoaderLogger = logging.getLogger("data.Base.dataLoaders")


def get_bin_bounds_for_voice_densities(voice_counts_per_sample: list, num_nonzero_bins=3):
    """
    Calculates the lower and upper bounds for the voice density bins

    category 0: no hits_upcoming_playback


    :param voice_counts_per_sample:
    :param num_nonzero_bins:
    :return: lower_bounds, upper_bounds
    """

    assert num_nonzero_bins > 0, "num_nonzero_bins should be greater than 0"

    non_zero_counts = sorted([count for count in voice_counts_per_sample if count > 0])

    samples_per_bin = len(non_zero_counts) // num_nonzero_bins

    grouped_bins = [non_zero_counts[i * samples_per_bin: (i + 1) * samples_per_bin] for i in range(num_nonzero_bins)]

    lower_bounds = [group[0] for group in grouped_bins]
    upper_bounds = [group[-1] for group in grouped_bins]
    upper_bounds[-1] = non_zero_counts[-1] + 1

    return lower_bounds, upper_bounds


def map_voice_densities_to_categorical(voice_counts, lower_bounds, upper_bounds):
    """
    Maps the voice counts to a categorical value based on the lower and upper bounds provided
    :param voice_counts:
    :param lower_bounds:
    :param upper_bounds:
    :return:
    """

    categories = []
    adjusted_upper_bounds = upper_bounds.copy()
    adjusted_upper_bounds[-1] = adjusted_upper_bounds[
                                    -1] + 1  # to ensure that the last bin is inclusive on the upper bound

    for count in voice_counts:
        if count == 0:
            categories.append(0)
        else:
            for idx, (low, high) in enumerate(zip(lower_bounds, adjusted_upper_bounds)):
                if low <= count < high:
                    categories.append(idx + 1)
                    break

    return categories

def map_tempo_to_categorical(tempo, n_tempo_bins=6):
    """
    Maps the tempo to a categorical value based on the following bins:
    0-60, 60-76, 76-108, 108-120, 120-168, 168-Above
    :param tempo:
    :param n_tempo_bins: [int] number of tempo bins to use (default is 6 and only 6 is supported at the moment)
    :return:
    """
    if n_tempo_bins != 6:
        raise NotImplementedError("Only 6 bins are supported for tempo mapping at the moment")

    if tempo < 60:
        return 0
    elif 60 <= tempo < 76:
        return 1
    elif 76 <= tempo < 108:
        return 2
    elif 108 <= tempo < 120:
        return 3
    elif 120 <= tempo < 168:
        return 4
    elif 168 <= tempo:
        return 5

def map_global_density_to_categorical(total_hits, max_hits, n_global_density_bins=8):
    """
    hit increase per bin = max_hits / n_global_density_bins

    :param total_hits:
    :param lower_bounds:
    :param upper_bounds:
    :return:
    """
    assert False, "This function is not used in the current implementation"

    step_res = max_hits / n_global_density_bins
    categories = []
    categories = [int(count / step_res) for count in total_hits]


    return categories

def map_value_to_bins(value, edges):
    """
    Maps a value to a bin based on the edges provided
    :param value:
    :param edges:
    :return:
    """
    for i in range(len(edges)+1):
        if i == 0:
            if value < edges[i]:
                return i
        elif i == len(edges):
            if value >= edges[-1]:
                return i
        else:
            if edges[i - 1] <= value < edges[i]:
                return i

    print("SHOULD NOT REACH HERE")

def map_drum_to_groove_hit_ratio_to_categorical(hit_ratios):
    # check bottomn of the file for the bin calculation
    _10_bins = [1.149999976158142, 1.2666666507720947, 1.3333333730697632, 1.4137930870056152, 1.4800000190734863,
                1.5357142686843872, 1.615384578704834, 1.7142857313156128, 1.8666666746139526]

    categories = []
    for hit_ratio in hit_ratios:
        categories.append(map_value_to_bins(hit_ratio, _10_bins))
    return categories

def load_bz2_hvo_sequences(dataset_setting_json_path, subset_tag, force_regenerate=False):
    """
    Loads the hvo_sequences using the settings provided in the json file.

    :param dataset_setting_json_path: path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
    :param subset_tag: [str] whether to load the train/test/validation set
    :param force_regenerate:
    :return:
    a list of hvo_sequences loaded from all the datasets specified in the json file
    """

    # load settings
    dataset_setting_json = json.load(open(dataset_setting_json_path, "r"))

    # load datasets
    dataset_tags = [key for key in dataset_setting_json["settings"].keys()]

    loaded_samples = []

    for dataset_tag in dataset_tags:
        dataLoaderLogger.info(f"Loading {dataset_tag} dataset")
        raw_data_pickle_path = dataset_setting_json["raw_data_pickle_path"][dataset_tag]

        for path_prepend in ["./", "../", "../../"]:
            if os.path.exists(path_prepend + raw_data_pickle_path):
                raw_data_pickle_path = path_prepend + raw_data_pickle_path
                break
        assert os.path.exists(raw_data_pickle_path), "path to gmd dict pickle is incorrect --- " \
                                                "look into data/***/storedDicts/groove-*.bz2pickle"

        dir__ = get_data_directory_using_filters(dataset_tag, dataset_setting_json_path)
        beat_division_factor = dataset_setting_json["global"]["beat_division_factor"]
        drum_mapping_label = dataset_setting_json["global"]["drum_mapping_label"]

        if (not os.path.exists(dir__)) or force_regenerate is True:
            dataLoaderLogger.info(f"load_bz2_hvo_sequences() --> No Cached Version Available Here: {dir__}. ")
            dataLoaderLogger.info(
                f"extracting data from raw pickled midi/note_sequence/metadata dictionaries at {raw_data_pickle_path}")
            gmd_dict = load_original_gmd_dataset_pickle(raw_data_pickle_path)
            drum_mapping = get_drum_mapping_using_label(drum_mapping_label)
            hvo_dict = extract_hvo_sequences_dict(gmd_dict, beat_division_factor, drum_mapping)
            pickle_hvo_dict(hvo_dict, dataset_tag, dataset_setting_json_path)
            dataLoaderLogger.info(f"load_bz2_hvo_sequences() --> Cached Version available at {dir__}")
        else:
            dataLoaderLogger.info(f"load_bz2_hvo_sequences() --> Loading Cached Version from: {dir__}")

        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        data = pickle.load(ifile)
        ifile.close()
        loaded_samples.extend(data)

    return loaded_samples


def collect_train_set_info(dataset_setting_json_path_, num_voice_density_bins, num_global_density_bins, max_len=32):
    """

    :param dataset_setting_json_path_:
    :param num_voice_density_bins:
    :param num_global_density_bins:
    :return:
     (kick_low_bound, kick_up_bound), (snare_low_bound, snare_up_bound), (hat_low_bound, hat_up_bound),
        (tom_low_bound, tom_up_bound), (cymbal_low_bound, cymbal_up_bound),
        (global_density_low_bound, global_density_up_bound), (complexity_low_bound, complexity_up_bound), genre_tags
    """
    train_set_genre_tags = []
    train_set_complexities = []
    train_set_kick_counts = []
    train_set_snare_counts = []
    train_set_hat_counts = []
    train_set_tom_counts = []
    train_set_cymbal_counts = []
    train_set_total_hits = []
    train_set_hvo_files = []
    training_set_ = load_bz2_hvo_sequences(dataset_setting_json_path_, "train", force_regenerate=False)

    for ix, hvo_sample in enumerate(
            tqdm(training_set_,
                 desc="collecting genre tags and Per Voice Density Bins from corresponding full TRAINING set")):
        hits = hvo_sample.hits_upcoming_playback
        if hits is not None:
            train_set_hvo_files.append(hvo_sample.metadata["full_midi_filename"])
            hits = hvo_sample.hvo[:, :9]
            if hits.sum() > 0:
                hvo_sample.adjust_length(max_len)
                if hvo_sample.metadata["style_primary"] not in train_set_genre_tags:  # collect genre tags from training set
                    train_set_genre_tags.append(hvo_sample.metadata["style_primary"])
                train_set_complexities.append(
                    hvo_sample.get_complexity_surprisal()[0])  # collect complexity surprisal from training set
                train_set_total_hits.append(hits.sum())
                train_set_kick_counts.append(hits[:, 0].sum())
                train_set_snare_counts.append(hits[:, 1].sum())
                train_set_hat_counts.append(hits[:, 2:4].sum())
                train_set_tom_counts.append(hits[:, 4:7].sum())
                train_set_cymbal_counts.append(hits[:, 7:].sum())

    # get pervoice density bins
    return (get_bin_bounds_for_voice_densities(train_set_kick_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_snare_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_hat_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_tom_counts, num_voice_density_bins),
            get_bin_bounds_for_voice_densities(train_set_cymbal_counts, num_voice_density_bins),
            None,
            (min(train_set_complexities), max(train_set_complexities)), sorted(train_set_genre_tags),
            train_set_total_hits, train_set_hvo_files)


# ---------------------------------------------------------------------------------------------- #
# loading a down sampled dataset
# ---------------------------------------------------------------------------------------------- #

def down_sample_mega_dataset(hvo_seq_list, down_sample_ratio):
    """
    Down sample the dataset by a given ratio, it does it per style
    :param hvo_seq_list:
    :param down_sample_ratio:
    :return:
    """

    down_sampled_set = []

    per_style_hvos = dict()
    for hvo_seq in hvo_seq_list:
        style = hvo_seq.metadata["style_primary"]
        if style not in per_style_hvos:
            per_style_hvos[style] = []
        per_style_hvos[style].append(hvo_seq)

    total_down_sampled_size = ceil(len(hvo_seq_list) * down_sample_ratio)
    # ensure that equal number of samples are taken from each style
    per_style_size = int(total_down_sampled_size / len(per_style_hvos))

    for style, hvo_seqs in per_style_hvos.items():
        if per_style_size > 0:
            size_ = min(per_style_size, len(hvo_seqs))
            down_sampled_set.extend(random.sample(hvo_seqs, size_))

    return down_sampled_set


def down_sample_gmd_dataset(hvo_seq_list, down_sample_ratio):
    """
    Down sample the dataset by a given ratio, the ratio of the performers and the ratio of the performances
    are kept the same as much as possible.
    :param hvo_seq_list:
    :param down_sample_ratio:
    :return:
    """
    down_sampled_size = ceil(len(hvo_seq_list) * down_sample_ratio)

    # =================================================================================================
    # Divide up the performances by performer
    # =================================================================================================
    per_performer_per_performance_data = dict()
    for hs in tqdm(hvo_seq_list):
        performer = hs.metadata["drummer"]
        performance_id = hs.metadata["master_id"]
        if performer not in per_performer_per_performance_data:
            per_performer_per_performance_data[performer] = {}
        if performance_id not in per_performer_per_performance_data[performer]:
            per_performer_per_performance_data[performer][performance_id] = []
        per_performer_per_performance_data[performer][performance_id].append(hs)

    # =================================================================================================
    # Figure out how many loops to grab from each performer
    # =================================================================================================
    def flatten(l):
        if isinstance(l[0], list):
            return [item for sublist in l for item in sublist]
        else:
            return l

    ratios_to_other_performers = dict()

    # All samples per performer
    existing_sample_ratios_by_performer = dict()
    for performer, performances in per_performer_per_performance_data.items():
        existing_sample_ratios_by_performer[performer] = \
            len(flatten([performances[p] for p in performances])) / len(hvo_seq_list)

    new_samples_per_performer = dict()
    for performer, ratio in existing_sample_ratios_by_performer.items():
        samples = ceil(down_sampled_size * ratio)
        if samples > 0:
            new_samples_per_performer[performer] = samples

    # =================================================================================================
    # Figure out for each performer, how many samples to grab from each performance
    # =================================================================================================
    num_loops_from_each_performance_compiled_for_all_performers = dict()
    for performer, performances in per_performer_per_performance_data.items():
        total_samples = len(flatten([performances[p] for p in performances]))
        if performer in new_samples_per_performer:
            needed_samples = new_samples_per_performer[performer]
            num_loops_from_each_performance = dict()
            for performance_id, hs_list in performances.items():
                samples_to_select = ceil(needed_samples * len(hs_list) / total_samples)
                if samples_to_select > 0:
                    num_loops_from_each_performance[performance_id] = samples_to_select
            if num_loops_from_each_performance:
                num_loops_from_each_performance_compiled_for_all_performers[performer] = \
                    num_loops_from_each_performance

    # =================================================================================================
    # Sample required number of loops from each performance
    # THE SELECTION IS DONE BY RANKING THE TOTAL NUMBER OF HITS / TOTAL NUMBER OF VOICES ACTIVE
    # then selecting N equally spaced samples from the ranked list
    # =================================================================================================
    for performer, performances in per_performer_per_performance_data.items():
        for performance, hvo_seqs in performances.items():
            seqs_sorted = sorted(
                hvo_seqs,
                key=lambda x: x.hits_upcoming_playback.sum() / x.get_number_of_active_voices(), reverse=True)
            indices = np.linspace(
                0,
                len(seqs_sorted) - 1,
                num_loops_from_each_performance_compiled_for_all_performers[performer][performance],
                dtype=int)
            per_performer_per_performance_data[performer][performance] = [seqs_sorted[i] for i in indices]

    downsampled_hvo_sequences = []
    for performer, performances in per_performer_per_performance_data.items():
        for performance, hvo_seqs in performances.items():
            downsampled_hvo_sequences.extend(hvo_seqs)

    return downsampled_hvo_sequences


def up_sample_to_ensure_genre_balance(hvo_seq_list):
    """
    Upsamples the dataset to ensure genre balance. Repeats the samples from each genre to match the size of the largest
    genre.

    :param hvo_seq_list:
    :return:
    """
    hvo_seq_per_genre = dict()
    for hvo_seq in hvo_seq_list:
        genre = hvo_seq.metadata["style_primary"]
        if genre not in hvo_seq_per_genre:
            hvo_seq_per_genre[genre] = []
        hvo_seq_per_genre[genre].append(hvo_seq)

    max_genre_size = max([len(hvo_seq_per_genre[genre]) for genre in hvo_seq_per_genre.keys()])
    upsampled_hvo_sequences = []

    for genre, hvo_seqs in hvo_seq_per_genre.items():
        # get number of repeats
        num_repeats = ceil(max_genre_size / len(hvo_seqs))
        tmp = hvo_seqs * num_repeats
        upsampled_hvo_sequences.extend(tmp[:max_genre_size])

    return upsampled_hvo_sequences

def load_downsampled_mega_set_hvo_sequences(
        dataset_setting_json_path, subset_tag, down_sampled_ratio, cache_down_sampled_set=True, force_regenerate=False):
    """
    Loads the hvo_sequences using the settings provided in the json file.

    :param dataset_setting_json_path: path to the json file containing the dataset settings (see data/dataset_json_settings/4_4_Beats_gmd.json)
    :param subset_tag: [str] whether to load the train/test/validation set
    :param down_sampled_ratio: [float] the ratio of the dataset to downsample to
    :param cache_downsampled_set: [bool] whether to cache the down sampled dataset
    :param force_regenerate: [bool] if True, will regenerate the hvo_sequences from the raw data regardless of cache
    :return:
    """
    dataset_tag = "mega_set"
    dir__ = get_data_directory_using_filters(dataset_tag,
                                             dataset_setting_json_path,
                                             down_sampled_ratio=down_sampled_ratio)
    if (not os.path.exists(dir__)) or force_regenerate is True or cache_down_sampled_set is False:
        dataLoaderLogger.info(f"No Cached Version Available Here: {dir__}. ")
        dataLoaderLogger.info(f"Downsampling the dataset to {down_sampled_ratio} and saving to {dir__}.")

        down_sampled_dict = {}
        for subset_tag in ["train", "validation", "test"]:
            hvo_seq_set = load_bz2_hvo_sequences(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_tag,
                force_regenerate=False)
            if down_sampled_ratio is not None:
                down_sampled_dict.update({subset_tag: down_sample_mega_dataset(hvo_seq_set, down_sampled_ratio)})
            else:
                down_sampled_dict.update({subset_tag: hvo_seq_set})

        # collect and dump samples that match filter
        if cache_down_sampled_set:
            # create directories if needed
            if not os.path.exists(dir__):
                os.makedirs(dir__)
            for set_key_, set_data_ in down_sampled_dict.items():
                ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
                pickle.dump(set_data_, ofile)
                ofile.close()

        dataLoaderLogger.info(f"Loaded {len(down_sampled_dict[subset_tag])} {subset_tag} samples from {dir__}")
        return down_sampled_dict[subset_tag]
    else:
        dataLoaderLogger.info(f"load_downsampled_mega_set_hvo_sequences() -> Loading Cached Version from: {dir__}")
        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        set_data_ = pickle.load(ifile)
        ifile.close()
        dataLoaderLogger.info(f"Loaded {len(set_data_)} {subset_tag} samples from {dir__}")
        return set_data_

def upsample_to_ensure_genre_balance(dataset_setting_json_path, subset_tag, cache_upsampled_set=True, force_regenerate=False):
    dataset_tag = list(json.load(open(dataset_setting_json_path, "r"))["settings"].keys())[0]
    dir__ = get_data_directory_using_filters(dataset_tag,
                                             dataset_setting_json_path,
                                             up_sampled_ratio=1)
    if (not os.path.exists(dir__)) or force_regenerate is True or cache_upsampled_set is False:
        dataLoaderLogger.info(f"No Cached Version Available Here: {dir__}. ")
        dataLoaderLogger.info(f"Upsampling the dataset to ensure genre balance and saving to {dir__}.")

        up_sampled_dict = {}
        for subset_tag in ["train", "validation", "test"]:
            hvo_seq_set = load_bz2_hvo_sequences(
                dataset_setting_json_path=dataset_setting_json_path,
                subset_tag=subset_tag,
                force_regenerate=False)
            up_sampled_dict.update({subset_tag: up_sample_to_ensure_genre_balance(hvo_seq_set)})

        # collect and dump samples that match filter
        if cache_upsampled_set:
            # create directories if needed
            if not os.path.exists(dir__):
                os.makedirs(dir__)
            for set_key_, set_data_ in up_sampled_dict.items():
                ofile = bz2.BZ2File(os.path.join(dir__, f"{set_key_}.bz2pickle"), 'wb')
                pickle.dump(set_data_, ofile)
                ofile.close()

        dataLoaderLogger.info(f"Loaded {len(up_sampled_dict[subset_tag])} {subset_tag} samples from {dir__}")
        return up_sampled_dict[subset_tag]
    else:
        dataLoaderLogger.info(f"upsample_to_ensure_genre_balance() -> Loading Cached Version from: {dir__}")
        ifile = bz2.BZ2File(os.path.join(dir__, f"{subset_tag}.bz2pickle"), 'rb')
        set_data_ = pickle.load(ifile)
        ifile.close()
        dataLoaderLogger.info(f"Loaded {len(set_data_)} {subset_tag} samples from {dir__}")
        return set_data_


class PairedLTADataset(Dataset):
    def __init__(self,
                 input_inst_dataset_bz2_filepath,
                 output_inst_dataset_bz2_filepath,
                 shift_tgt_by_n_steps=1,
                 max_input_bars=32,
                 continuation_bars=2,
                 hop_n_bars=2,
                 input_has_velocity=True,
                 input_has_offsets=True,
                 push_all_data_to_cuda=False):

        self.max_input_bars = max_input_bars
        self.continuation_bars = continuation_bars
        self.hop_n_bars = hop_n_bars
        self.input_has_velocity = input_has_velocity
        self.input_has_offsets = input_has_offsets

        def get_cached_filepath():
            dir_ = "cached/TorchDatasets"
            filename = (
                f"PairedLTADataset_{input_inst_dataset_bz2_filepath.split('/')[-1]}_{output_inst_dataset_bz2_filepath.split('/')[-1]}"
                f"_{max_input_bars}_{hop_n_bars}_{input_has_velocity}_{input_has_offsets}_{shift_tgt_by_n_steps}.bz2pickle")
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            return os.path.join(dir_, filename)

        def stack_two_hvos(hvo1, hvo2, use_velocity, use_offsets):
            # assert same length
            assert hvo1.shape[0] == hvo2.shape[0], f"{hvo1.shape} != {hvo2.shape}"

            n_voices_1 = hvo1.shape[-1] // 3
            n_voices_2 = hvo2.shape[-1] // 3

            h1 = hvo1[:, :n_voices_1]
            v1 = hvo1[:, n_voices_1:2 * n_voices_1]
            o1 = hvo1[:, 2 * n_voices_1:]
            h2 = hvo2[:, :n_voices_2]
            v2 = hvo2[:, n_voices_2:2 * n_voices_2]
            o2 = hvo2[:, 2 * n_voices_2:]

            if use_velocity and use_offsets:
                return np.hstack([h1, h2, v1, v2, o1, o2])
            elif use_offsets:
                return np.hstack([h1, h2, o1, o2])
            elif use_velocity:
                return np.hstack([h1, h2, v1, v2])
            else:
                return np.hstack([h1, h2])

        # check if cached version exists
        cached_exists = os.path.exists(get_cached_filepath())

        if cached_exists:
            # todo load cached version
            should_process_data = False

            # Load the cached version
            dataLoaderLogger.info(f"PairedLTADatasetV2 Constructor --> Loading Cached Version from: {get_cached_filepath()}")
            ifile = bz2.BZ2File(get_cached_filepath(), 'rb')
            data = pickle.load(ifile)
            ifile.close()
            self.instrument1_hvos = torch.tensor(data["instrument1_hvos"], dtype=torch.float32)
            self.instrument2_hvos = torch.tensor(data["instrument2_hvos"], dtype=torch.float32)
            self.instrument1and2_hvos = torch.tensor(data["instrument1and2_hvos"], dtype=torch.float32)

        else:
            should_process_data = True

        if should_process_data:
            # load data
            with bz2.BZ2File(input_inst_dataset_bz2_filepath, 'rb') as f:
                instrument1s = pickle.load(f)
            with bz2.BZ2File(output_inst_dataset_bz2_filepath, 'rb') as f:
                instrument2s = pickle.load(f)

            # get song ids that are common in both datasets
            song_ids_all = list(set(instrument1s.keys()) & set(instrument2s.keys()))
            song_ids_ = []

            inst1_hvos = []
            inst2_hvos = []
            inst1and2_hvos = []

            max_segment_len_bars = max_input_bars + continuation_bars

            for song_id in tqdm(song_ids_all):
                i1 = instrument1s[song_id]
                i2 = instrument2s[song_id]

                if len(i1.time_signatures) == 1 and len(i2.time_signatures) == 1:
                    if (i1.time_signatures[0].numerator == i2.time_signatures[0].numerator and
                            i1.time_signatures[0].denominator == i2.time_signatures[0].denominator):
                        if i1.time_signatures[0].numerator == 4 and i1.time_signatures[0].denominator == 4:
                            song_ids_.append(song_id)

                # adjust to the same length
                n_steps = max(i1.hvo.shape[0], i2.hvo.shape[0])
                i1.adjust_length(n_steps)
                i2.adjust_length(n_steps)

                n_bars = n_steps // 16

                segments_starts = [i for i in range(0, n_bars, hop_n_bars)]

                for i in segments_starts:
                    ts_ = i * 16
                    te_ = ts_ + max_segment_len_bars * 16
                    i1_seg = i1.hvo[ts_:te_]
                    i2_seg = i2.hvo[ts_:te_]

                    if i1_seg.shape[0] == max_segment_len_bars * 16:
                        i1_2_stack = stack_two_hvos(i1_seg, i2_seg, input_has_velocity, input_has_offsets)
                        inst1_hvos.append(i1_seg)
                        inst2_hvos.append(i2_seg)
                        inst1and2_hvos.append(i1_2_stack)

            self.instrument1_hvos = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in inst1_hvos])
            self.instrument2_hvos = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in inst2_hvos])
            self.instrument1and2_hvos = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in inst1and2_hvos])

            # cache the processed data
            data_to_dump = {
                "instrument1_hvos": self.instrument1_hvos.numpy(),
                "instrument2_hvos": self.instrument2_hvos.numpy(),
                "instrument1and2_hvos": self.instrument1and2_hvos.numpy(),
            }

            ofile = bz2.BZ2File(get_cached_filepath(), 'wb')
            pickle.dump(data_to_dump, ofile)

            ofile.close()

            dataLoaderLogger.info(f"Loaded {len(self.instrument1_hvos)} sequences")

        if push_all_data_to_cuda:
            self.instrument1_hvos = self.instrument1_hvos.cuda()
            self.instrument2_hvos = self.instrument2_hvos.cuda()
            self.instrument1and2_hvos = self.instrument1and2_hvos.cuda()

    def get_hit_density_histogram(self, n_bins=100):
        hit_densities = []
        for i in range(len(self)):
            _, _, i12 = self[i]
            hit_densities.append(i12[:, :10].numpy().flatten().mean())

        hit_density_hist, bin_edges, _ = binned_statistic(hit_densities, hit_densities, statistic='count', bins=n_bins)
        return hit_density_hist, bin_edges

    def show_hit_density_histogram(self, n_bins=100):
        hit_density_hist, bin_edges = self.get_hit_density_histogram(n_bins)
        plt.bar(bin_edges[:-1], hit_density_hist, width=bin_edges[1] - bin_edges[0])
        plt.show()

    def __len__(self):
        return len(self.instrument1_hvos)

    def __getitem__(self, idx):
        return (self.instrument1_hvos[idx],  # 0: instrument1_hvos (shape: (max_input_bars + continuation_bars, 3 * features_inst_1))
                self.instrument2_hvos[idx],  # 1: instrument2_hvos (shape: (max_input_bars + continuation_bars, 3 * features_inst_2))
                self.instrument1and2_hvos[idx]  # 2: instrument1and2_hvos (shape: (max_input_bars + continuation_bars, 3 * features_inst_1 + 3 * features_inst_2))
                )


class StackedLTADatasetV2(Dataset):
    def __init__(self,
                 input_inst_dataset_bz2_filepath,
                 output_inst_dataset_bz2_filepath,
                 shift_tgt_by_n_steps=1,
                 max_input_bars=32,
                 hop_n_bars=2,
                 push_all_data_to_cuda=False,
                 input_has_velocity=True):

        self.max_input_bars = max_input_bars
        self.hop_n_bars = hop_n_bars

        def get_cached_filepath():
            dir_ = "cached/TorchDatasets"
            filename = (
                f"StackedLTADatasetV2_{input_inst_dataset_bz2_filepath.split('/')[-1]}_{output_inst_dataset_bz2_filepath.split('/')[-1]}"
                f"_{max_input_bars}_{hop_n_bars}_{shift_tgt_by_n_steps}.bz2pickle")
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            return os.path.join(dir_, filename)

        def stack_two_hvos(hvo1, hvo2, use_velocity):
            # assert same length
            assert hvo1.shape[0] == hvo2.shape[0], f"{hvo1.shape} != {hvo2.shape}"

            n_voices_1 = hvo1.shape[-1] // 3
            n_voices_2 = hvo2.shape[-1] // 3

            h1 = hvo1[:, :n_voices_1]
            v1 = hvo1[:, n_voices_1:2 * n_voices_1]
            o1 = hvo1[:, 2 * n_voices_1:]
            h2 = hvo2[:, :n_voices_2]
            v2 = hvo2[:, n_voices_2:2 * n_voices_2]
            o2 = hvo2[:, 2 * n_voices_2:]

            if use_velocity:
                return np.hstack([h1, h2, v1, v2, o1, o2])
            else:
                return np.hstack([h1, h2, o1, o2])

        # check if cached version exists
        cached_exists = os.path.exists(get_cached_filepath())

        if cached_exists:
            # todo load cached version
            should_process_data = False

            # Load the cached version
            dataLoaderLogger.info(f"StackedLTADatasetV2 Constructor --> Loading Cached Version from: {get_cached_filepath()}")
            ifile = bz2.BZ2File(get_cached_filepath(), 'rb')
            data = pickle.load(ifile)
            ifile.close()
            self.instrument1_hvos = torch.tensor(data["instrument1_hvos"], dtype=torch.float32)
            self.instrument2_hvos = torch.tensor(data["instrument2_hvos"], dtype=torch.float32)
            self.stacked_target = torch.tensor(data["stacked_target"], dtype=torch.float32)
            self.stacked_target_shifted = torch.tensor(data["stacked_target_shifted"], dtype=torch.float32)
        else:
            dataLoaderLogger.info(f"No Cached Version Available Here: {get_cached_filepath}. ")
            should_process_data = True

        def get_hit_counts_per_bar(hvo_in):
            counts = []
            n_features = hvo_in.shape[-1] // 3
            for i in range(0, hvo_in.shape[0], 16):
                counts.append(hvo_in[i:i+16, :n_features].sum())
            return counts

        def is_valid(hvo_groove, hvo_drum):
            groove_counts = get_hit_counts_per_bar(hvo_groove)
            drum_counts = get_hit_counts_per_bar(hvo_drum)

            # invalid if first groove bar has no hits or drum hits
            # invalid if more than 2 empty groove bars or more than 2 empty drum bars
            if groove_counts[0] == 0:
                return False
            elif groove_counts.count(0) > 2 or drum_counts.count(0) > 3:
                return False
            else:
                return True

        if should_process_data:
            # load data
            with bz2.BZ2File(input_inst_dataset_bz2_filepath, 'rb') as f:
                instrument1s = pickle.load(f)
            with bz2.BZ2File(output_inst_dataset_bz2_filepath, 'rb') as f:
                instrument2s = pickle.load(f)

            # get song ids that are common in both datasets
            song_ids_all = list(set(instrument1s.keys()) & set(instrument2s.keys()))
            song_ids_ = []

            inst1_hvos = []
            inst2_hvos = []
            stacked_target = []
            stacked_target_shifted = []

            for song_id in tqdm(song_ids_all):
                i1 = instrument1s[song_id]
                i2 = instrument2s[song_id]

                if len(i1.time_signatures) == 1 and len(i2.time_signatures) == 1:
                    if (i1.time_signatures[0].numerator == i2.time_signatures[0].numerator and
                            i1.time_signatures[0].denominator == i2.time_signatures[0].denominator):
                        if i1.time_signatures[0].numerator == 4 and i1.time_signatures[0].denominator == 4:
                            song_ids_.append(song_id)

                # adjust to the same length
                n_steps = max(i1.hvo.shape[0], i2.hvo.shape[0])
                i1.adjust_length(n_steps)
                i2.adjust_length(n_steps)

                n_bars = n_steps // 16

                segments_starts = [i for i in range(0, n_bars, hop_n_bars)]
                for i in segments_starts:
                    ts_ = i * 16
                    te_ = ts_ + max_input_bars * 16
                    i1_seg = i1.hvo[ts_:te_].copy()
                    i2_seg = i2.hvo[ts_:te_].copy()

                    # mute first bar of instrument 2 (drums)
                    i2_seg[:16] = 0

                    if i1_seg.shape[0] == max_input_bars * 16 and is_valid(i1_seg, i2_seg):
                        i1_2_stack = stack_two_hvos(i1_seg, i2_seg, True)
                        inst1_hvos.append(i1_seg)
                        inst2_hvos.append(i2_seg)
                        stacked_target.append(i1_2_stack)
                        i1_2_stack_for_shifting = stack_two_hvos(i1_seg, i2_seg, input_has_velocity)
                        stacked_target_shifted_ = np.zeros_like(i1_2_stack_for_shifting)
                        stacked_target_shifted_[shift_tgt_by_n_steps:] = i1_2_stack_for_shifting[:-shift_tgt_by_n_steps]
                        stacked_target_shifted.append(stacked_target_shifted_)

            self.instrument1_hvos = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in inst1_hvos])
            self.instrument2_hvos = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in inst2_hvos])
            self.stacked_target = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in stacked_target])
            self.stacked_target_shifted = torch.vstack([torch.tensor(x, dtype=torch.float32).unsqueeze(0) for x in stacked_target_shifted])


            # cache the processed data
            data_to_dump = {
                "instrument1_hvos": self.instrument1_hvos.numpy(),
                "instrument2_hvos": self.instrument2_hvos.numpy(),
                "stacked_target": self.stacked_target.numpy(),
                "stacked_target_shifted": self.stacked_target_shifted.numpy(),
            }

            ofile = bz2.BZ2File(get_cached_filepath(), 'wb')
            pickle.dump(data_to_dump, ofile)

            ofile.close()

            dataLoaderLogger.info(f"Loaded {len(self.instrument1_hvos)} sequences")

        if push_all_data_to_cuda:
            self.stacked_target = self.stacked_target.cuda()
            self.stacked_target_shifted = self.stacked_target_shifted.cuda()

    def get_hit_density_histogram(self, n_bins=100):
        hit_densities = []
        for i in range(len(self)):
            _, _, i12 = self[i]
            hit_densities.append(i12[:, :10].numpy().flatten().mean())

        hit_density_hist, bin_edges, _ = binned_statistic(hit_densities, hit_densities, statistic='count', bins=n_bins)
        return hit_density_hist, bin_edges

    def show_hit_density_histogram(self, n_bins=100):
        hit_density_hist, bin_edges = self.get_hit_density_histogram(n_bins)
        plt.bar(bin_edges[:-1], hit_density_hist, width=bin_edges[1] - bin_edges[0])
        plt.show()

    def __len__(self):
        return len(self.instrument1_hvos)

    def __getitem__(self, idx):
        return (
            self.stacked_target_shifted[idx],
            self.stacked_target[idx],
            self.instrument1_hvos[idx],
            self.instrument2_hvos[idx]
                )


if __name__ == "__main__":
    # tester
    dataLoaderLogger.info("Run demos/data/demo.py to test")


    #
    # =================================================================================================
    # Load Mega dataset as torch.utils.data.Dataset


    def get_hit_counts_per_bar(hvo_in):
        counts = []
        n_features = hvo_in.shape[-1] // 3
        for i in range(0, hvo_in.shape[0], 16):
            counts.append(hvo_in[i:i + 16, :n_features].sum())
        return counts

    max_n_bars = 8

    test_dataset = StackedLTADatasetV2(
            input_inst_dataset_bz2_filepath="data/lmd/data_bass_groove_train.bz2",
            output_inst_dataset_bz2_filepath="data/lmd/data_drums_full_unsplit.bz2",
            shift_tgt_by_n_steps=1,
            max_input_bars=max_n_bars,
            hop_n_bars=max_n_bars,
            push_all_data_to_cuda=False,
            input_has_velocity=True)


    get_hit_counts_per_bar(test_dataset.instrument2_hvos[10])

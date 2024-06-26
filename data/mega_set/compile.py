from API_ import MidiCollection
import bz2
import pickle
import tqdm
import os
import zipfile


def dict_append(dictionary, key, vals):
    """Appends a value or a list of values to a key in a dictionary"""

    # if the values for a key are not a list, they are converted to a list and then extended with vals
    dictionary[key] = list(dictionary[key]) if not isinstance(dictionary[key], list) else dictionary[key]

    # if vals is a single value (not a list), it's converted to a list so as to be iterable
    vals = [vals] if not isinstance(vals, list) else vals

    # append new values
    for val in vals:
        dictionary[key].append(val)

    return dictionary

def get_as_dict(dataset_collection):
    """
    mixes tfds dataset and metadata into a dictionary

    :param dataset: DatasetV1Adapter (TFDS loaded dataset)
    :param beat_division_factors: list, default ([4])
    :param csv_dataframe_info: pandas.df (default None)
    :return:
    """

    dataset_dict_processed = dict()
    dataset_dict_processed.update({
        "drummer": [],
        "session": [],
        "loop_id": [],  # the id of the recording from which the loop is extracted
        "master_id": [],  # the id of the recording from which the loop is extracted
        "style_primary": [],
        "style_secondary": [],
        "bpm": [],
        "beat_type": [],
        "time_signature": [],
        "full_midi_filename": [],
        "full_audio_filename": [],
        "midi": [],
        "note_sequence": [],
    })

    dataset_length = len(dataset_collection)

    for midiMeta in tqdm.tqdm(dataset_collection, total=dataset_length):

        # Features to be extracted from the dataset

        note_sequence = midiMeta.note_sequence()

        if note_sequence.notes:  # ignore if no notes in note_sequence (i.e. empty 2 bar sequence)

            # Get the relevant series from the dataframe
            file_name = midiMeta.exact_path.split("/")[-1]

            # Update the dictionary associated with the metadata
            dict_append(dataset_dict_processed, "drummer", midiMeta.collection)
            dict_append(dataset_dict_processed, "session", midiMeta.exact_path.split("/")[-2])
            dict_append(dataset_dict_processed, "loop_id", file_name.split("__")[-1])
            dict_append(dataset_dict_processed, "master_id", "__".join(file_name.split("__")[:-1]))

            style_primary = midiMeta.genre
            style_secondary = midiMeta.all_styles
            dict_append(dataset_dict_processed, "style_primary", style_primary)
            dict_append(dataset_dict_processed, "style_secondary", style_secondary)
            dict_append(dataset_dict_processed, "bpm", note_sequence.tempos[0].qpm)
            dict_append(dataset_dict_processed, "beat_type", midiMeta.type)
            dict_append(dataset_dict_processed, "time_signature", f"{note_sequence.time_signatures[0].numerator}-{note_sequence.time_signatures[0].denominator}")
            dict_append(dataset_dict_processed, "full_midi_filename", file_name)
            dict_append(dataset_dict_processed, "full_audio_filename", "")
            dict_append(dataset_dict_processed, "midi", "")
            dict_append(dataset_dict_processed, "note_sequence", [note_sequence])

    return dataset_dict_processed


def pickle_dict(gmd_dict, path, filename):

    if not os.path.exists(path):
        os.makedirs(path)

    ofile = bz2.BZ2File(os.path.join(path, filename+".bz2pickle"), 'wb')
    pickle.dump(gmd_dict, ofile)
    ofile.close()

if __name__ == "__main__":
    zip_path = 'MultiSets'
    # for zip_name in ['Balanced_100_per_genre_performed_4_4.zip',
    #                  'Balanced_500_per_genre_performed_4_4.zip',
    #                  'Balanced_1000_per_genre_performed_4_4.zip',
    #                  'Balanced_2000_per_genre_performed_4_4.zip',
    #                  'Balanced_3000_per_genre_performed_4_4.zip',
    #                  'Balanced_4000_per_genre_performed_4_4.zip',
    #                  'Balanced_5000_per_genre_performed_4_4.zip',
    #                  'Imbalanced_RockDownSampled.zip',
    for zip_name in ['About10000PerGenre_withRepeatsForBalancing.zip']:
        print("-"*50)
        print("Processing: ", zip_name)
        full_path = os.path.join(zip_path, zip_name)
        export_path = os.path.join(zip_path, zip_name.replace(".zip", "/"))
        print(export_path)

        # unzip
        with zipfile.ZipFile(full_path, 'r') as zip_ref:
            zip_ref.extractall(zip_path)

        # get paths
        train_csv = os.path.join(export_path, 'train/summary.csv')
        test_csv = os.path.join(export_path, 'test/summary.csv')
        validation_csv = os.path.join(export_path, 'val/summary.csv')

        # load Collections
        train_collection = MidiCollection.from_csv(train_csv)
        test_collection = MidiCollection.from_csv(test_csv)
        validation_collection = MidiCollection.from_csv(validation_csv)

        # convert to dict
        train_dict = get_as_dict(train_collection)
        test_dict = get_as_dict(test_collection)
        validation_dict = get_as_dict(validation_collection)

        # compile
        mega_dict = {"train": train_dict, "test": test_dict, "validation": validation_dict}

        # save
        pickle_dict(mega_dict, "storedDicts", zip_name.replace(".zip", ""))

        # delete extracted folder
        os.system(f"rm -rf {export_path}")
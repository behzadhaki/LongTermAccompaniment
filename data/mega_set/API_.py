import pandas
import pandas as pd
from collections import defaultdict
import note_seq
import os
import shutil
import numpy as np
import tqdm
from copy import deepcopy
import random

try:
    from hvo_sequence.hvo_seq import HVO_Sequence
    from hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING
    from hvo_sequence.io_helpers import midi_to_hvo_sequence
    _HAS_HVO_SEQ = True
except ImportError:
    _HAS_HVO_SEQ = False

style_mapping = {
    'rock': 'Rock',
    'blues': 'Blues',
    'punk': 'Rock',
    'rock+shuffle': 'Rock',
    'electronic+breakbeat+dnb+hiphop': 'Other', # : 'Electronic',
    'electronic': 'Other', # : 'Electronic',
    'blues+shuffle': 'Blues',
    'rnb': 'Hip-Hop/R&B/Soul',
    'funk': 'Funk',
    'shuffle+blues': 'Blues',
    'breakbeat+electronic': 'Other', # : 'Electronic',
    'country': 'Rock',
    'swing+blues': 'Blues',
    'rnb+swing': 'Hip-Hop/R&B/Soul',
    'dnb+electronic': 'Other', # : 'Electronic',
    'jazz': 'Jazz',
    'garage+electronic': 'Other', # : 'Electronic',
    'fusion+funk': 'Other',
    'world': 'Other',
    'electronic+rock': 'Other',
    'jungle+electronic': 'Other', # : 'Electronic',
    'hiphop': 'Hip-Hop/R&B/Soul',
    'electronic+dnb': 'Other', # : 'Electronic',
    'trance+electronic': 'Other', # : 'Electronic',
    'soul': 'Hip-Hop/R&B/Soul',
    'fusion': 'Other',
    'techno+electronic': 'Other', # : 'Electronic',
    'rock+progressive': 'Rock',
    'pop+euro+eurobeat': 'Pop',
    'electronic+trance': 'Other', # : 'Electronic',
    'house+electronic': 'Other', # : 'Electronic',
    'electronic+funk': 'Other', # : 'Electronic',
    'downtempo+electronic': 'Other', # : 'Electronic',
    'electronic+breakbeat': 'Other', # : 'Electronic',
    'punk+rock': 'Rock',
    'pop+rock': 'Pop',
    'rock+alternative': 'Rock',
    'rnb+swing+funk+shuffle': 'Hip-Hop/R&B/Soul',
    'indie+rock': 'Rock',
    'rnb+underground': 'Hip-Hop/R&B/Soul',
    'rnb+hiphop': 'Hip-Hop/R&B/Soul',
    'latin': 'Latin',
    'electronic+jungle': 'Other', # : 'Electronic',
    'progressive+rock': 'Rock',
    'rnb+swing+disco': 'Hip-Hop/R&B/Soul',
    'world+latin': 'Latin',
    'afro+world': 'Afro',
    'blues+rock': 'Blues',
    'electronic+garage': 'Other', # : 'Electronic',
    'jungle': 'Other', # : 'Electronic',
    'rock+funk': 'Other',
    'alternative+rock': 'Rock',
    'hiphop+rap': 'Hip-Hop/R&B/Soul',
    'highlife+world': 'Afro',
    'electronic+house': 'Other', # : 'Electronic',
    'bebop+hardbop': 'Jazz',
    'rnb+swing+soca': 'Hip-Hop/R&B/Soul',
    'electronic+dance': 'Other', # : 'Electronic',
    'electronic+techno': 'Other', # : 'Electronic',
    'swing+jazz': 'Jazz',
    'ballad+rock': 'Rock',
    'fusion+latin': 'Other',
    'afro+pop+world': 'Afro',
    'indie+disco': 'Disco',
    'rock+jazz+fusion': 'Other',
    'funk+rock': 'Other',
    'rock+country': 'Rock',
    'disco': 'Disco',
    'swing': 'Jazz',
    'triphop': 'Other', # : 'Electronic',
    'garage': 'Other', # : 'Electronic',
    'rumba+blues': 'Blues',
    'rock+jazz+shuffle': 'Other',
    'funk+shuffle': 'Other',
    'pop': 'Pop',
    'electronic+downtempo': 'Other', # : 'Electronic',
    'ballad+country': 'Rock',
    'electronic+breaks+swing': 'Other', # : 'Electronic',
    'hiphop+funk': 'Hip-Hop/R&B/Soul',
    'reggae+world': 'Reggae',
    'electronic+ballad': 'Other', # : 'Electronic',
    'soca+electronic+garage': 'Other', # : 'Electronic',
    'fusion+shuffle': 'Other',
    'fusion+ballad': 'Other',
    'country+ballad+shuffle': 'Rock',
    'electronic+pop+rock+dance': 'Other', # : 'Electronic',
    'electronic+swing': 'Other', # : 'Electronic',
    'electronic+reggae': 'Reggae',
    'electronic+afro': 'Afro',
    'bossanova+electronic': 'Other', # : 'Electronic',
    'shuffle+country': 'Rock',
    'electronic+samba+breakbeat+dnb+hiphop': 'Other', # : 'Electronic',
    'jazz+swing': 'Jazz',
    'electronic+fusion': 'Other',
    'house+funk': 'Other',
    'ballad+shuffle+country': 'Rock',
    'hiphop+funk+rap': 'Hip-Hop/R&B/Soul',
    'electronic+techno+dance': 'Other', # : 'Electronic',
    'shuffle+funk': 'Other',
    'fusion+jazz': 'Other',
    'electronic+jazz': 'Jazz',
    'shuffle+rnb': 'Hip-Hop/R&B/Soul',
    'jazz+breaks+fusion': 'Other',
    'afro+shuffle+world': 'Afro',
    'electronic+latin': 'Latin',
    'pop+rock+funk': 'Other',
    'electronic+pop+ballad+dance': 'Other', # : 'Electronic',
    'rnb+funk': 'Funk',
    'country+rock': 'Rock',
    'country+ballad': 'Rock',
    'rnb+pop': 'Hip-Hop/R&B/Soul',
    'rnb+ballad': 'Hip-Hop/R&B/Soul',
    'soul+funk': 'Funk',
    'neworleans+jazz': 'Jazz',
    'samba+fusion+latin': 'Latin',
    'rnb+motown': 'Hip-Hop/R&B/Soul',
    'funk+electronic+breakbeat': 'Other', # : 'Electronic',
    'country+shuffle': 'Rock',
    'rock+indie': 'Rock',
    'jazz+funk': 'Other',
    'breaks+fusion+funk': 'Other',
    'shuffle': 'Blues',
    'electronic+rock+breakbeat+dnb+hiphop': 'Other', # : 'Electronic',
    'latin+samba+funk+world': 'Latin',
    'jazz+electronic+breakbeat': 'Other', # : 'Electronic',
    'calypso+world': 'Latin',
    'bossanova+funk+world': 'Other',
    'bossanova+electronic+breakbeat+dnb+hiphop': 'Other', # : 'Electronic',
    'psychedelic+rock': 'Rock',
    'electronic+afro+breakbeat+dnb+hiphop': 'Hip-Hop/R&B/Soul',
    'afro': 'Afro',
    'swing+fusion+funk': 'Other',
    'funk+fusion+latin': 'Other',
    'electronic+reggae+breakbeat+dnb+hiphop': 'Hip-Hop/R&B/Soul',
    'breaks+jazz+fusion': 'Other',
    'traditional+pop': 'Pop',
    'rock+hardcore': 'Rock',
    'newwave+indie': 'Rock',
    'funk+jazz': 'Other',
    'latin+samba': 'Latin',
    'hiphop+electronic': 'Other', # : 'Electronic',
    'electronic+funk+jazz+breakbeat+dnb+hiphop': 'Other', # : 'Electronic',
    'ballad': 'Pop',
    'breaks+electronic': 'Other', # : 'Electronic',
    'latin+electronic+breakbeat': 'Other', # : 'Electronic',
    'breaks+electronic+garage': 'Other', # : 'Electronic',
    'blues+country': 'Blues',
    'soul+funk+smooth': 'Funk',
    'funk+triphop': 'Other',
    'ragtime': 'Jazz',
    'reggae+ska': 'Reggae',
    'swing+shuffle': 'Blues',
    'electronic+breaks+breakbeat': 'Other', # : 'Electronic',
    'latin+jazz+mambo': 'Latin',
    'dnb': 'Other', # : 'Electronic',
    'dance+jungle': 'Other', # : 'Electronic',
    'tribal': 'Other', # : 'Electronic',
    'pop+ballad': 'Pop',
    'latin+mambo': 'Latin',
    'bossanova': 'Latin',
    'soul+rnb': 'Hip-Hop/R&B/Soul',
    'reggae': 'Reggae',
    'latin+world+rock': 'Latin',
    'latin+country': 'Latin',
    'indie+rock+shuffle': 'Rock',
    'electronic+breakbeat+house': 'Other', # : 'Electronic',
    'ambient': 'Other', # : 'Electronic',
    'waltz': 'Jazz',
    'hiphop+soul+funk': 'Hip-Hop/R&B/Soul',
    'tribal+world': 'Other',
    'bossanova+latin+jazz': 'Latin',
    'bebop+traditional+pop': 'Jazz',
    'electronic+latin+trance': 'Other', # : 'Electronic',
    'electronic+jazz+breaks+swing': 'Other', # : 'Electronic',
    'electronic+industrial+swing': 'Other', # : 'Electronic',
    'blues+rock+shuffle': 'Blues',
    'dance': 'Other', # : 'Electronic',
    'bossanova+jazz+latin': 'Latin',
    'breaks+hiphop+electronic': 'Other', # : 'Electronic',
    'breakbeat+electronic+breaks+swing': 'Other', # : 'Electronic',
     'rock+ethno': 'Rock',
     'bebop+jazz': 'Jazz',
     'pop+country': 'Pop',
     'funk+swing': 'Funk',
     'electronic+ballad+breakbeat+dnb+hiphop': 'Other', # : 'Electronic',
     'pop+shuffle': 'Pop',
     'jazz+smooth': 'Jazz',
     'rock+blues': 'Blues',
     'soul+shuffle': 'Hip-Hop/R&B/Soul',
     'indie+rock+dance': 'Rock',
     'ethno': 'Other',
     'mambo+jazz+swing+latin': 'Latin',
     'pop+swing': 'Pop',
     'alternative+rock+ballad': 'Rock',
     'soul+rnb+ballad': 'Hip-Hop/R&B/Soul',
     'breaks+rock': 'Rock',
     'fusion+breaks+latin': 'Other',
     'afrocuban': 'Latin',
     'rock+pop': 'Rock',
     'ska+rock+punk': 'Rock',
     'breaks+jazz+smooth': 'Jazz',
     'latin+world+mambo': 'Latin',
     'electronic+smooth+breaks+swing': 'Other', # : 'Electronic',
     'dnb+breakbeat': 'Other', # : 'Electronic',
     'breaks+blues+shuffle': 'Blues',
     'mambo+jazz+funk': 'Other',
     'funk+jazz+fusion': 'Other',
     'funk+latin': 'Funk',
     'neworleans': 'Jazz',
     'bossanova+world': 'Latin',
     'bebop': 'Jazz',
     'newwave+indie+shuffle': 'Rock',
     'latin+world': 'Latin',
     'shuffle+rock+blues': 'Blues',
     'electronic+samba+breaks+swing': 'Other', # : 'Electronic',
     'breaks': 'Other', # : 'Electronic',
     'samba+jazz+latin': 'Latin',
     'latin+fusion': 'Latin',
     'blues+rumba': 'Blues',
     'funk+house': 'Other', # : 'Electronic',
     'jazz+ballad': 'Jazz',
     'rock+swing': 'Rock',
     'latin+world+rumba': 'Latin',
     'samba+fusion+breaks+latin': 'Latin',
     'blues+rock+ballad': 'Blues',
     'afrocuban+afro+world+funk': 'Latin',
     'jazz+latin': 'Jazz',
     'ballad+world': 'Pop',
     'calypso+jazz': 'Reggae',
     'breaks+punk': 'Rock',
     'cinematic': 'Other', # : 'Electronic',
     'indie+rock+tribal': 'Rock',
     'blues+funk': 'Funk',
     'bossanova+jazz': 'Latin',
     'breaks+blues': 'Blues',
     'shuffle+jazz+blues': 'Jazz',
     'latin+rumba+world': 'Latin',
     'rock+indie+shuffle': 'Rock',
     'afrobeat': 'Afro',
     'rock+indie+swing+funk': 'Rock',
     'dance+funk': 'Funk',
     'tribal+swing': 'Other', # : 'Electronic',
     'soul+ballad': 'Hip-Hop/R&B/Soul',
     'latin+mambo+world': 'Latin',
     'rock+indie+tribal+shuffle': 'Rock',
     'latin+jazz+rumba': 'Latin',
     'ska+world': 'Other',
     'mambo+jazz+latin': 'Latin',
     'blues+jazz': 'Jazz',
     'dancehall+dance+world': 'Reggae',
     'psychedelic+rock+rap': 'Rock',
     'shuffle+fusion': 'Other',
     'latin+jazz': 'Jazz',
     'latin+samba+world': 'Latin',
     'rock+punk': 'Rock',
     'country+breakbeat': 'Rock',
     'rock+indie+disco': 'Disco',
     'funk+garage': 'Other', # : 'Electronic',
     'jazz+fusion': 'Other',
     'rock+ballad': 'Rock',
     'jazz+blues': 'Jazz',
     'house': 'Other', # : 'Electronic',
     'rock+garage': 'Rock',
     'soul+funk+swing': 'Funk',
     'mambo+jazz+smooth': 'Other',
     'reggae+latin': 'Reggae',
     'latin+garage': 'Latin',
     'rock+indie+disco+smooth': 'Disco',
     'rock+indie+disco+dance': 'Disco',
     'funk+jazz+mambo': 'Other',
     'highlife': 'Afro',
}

class MidiMetadata:
    def __init__(self, length_group, num_measures, collection, time_signatures, tempo, is_performed,
                 genre, lev1_path, lev2_path, midi_file_name, is_fill, all_styles, type_, also_found_at,
                 root_dir, exact_path=None
                 ):
        self.length_group = length_group
        self.num_measures = num_measures
        self.collection = collection
        self.time_signatures = time_signatures
        self.tempo = tempo
        self.is_performed = is_performed
        self.genre = genre
        self.lev1_path = lev1_path
        self.lev2_path = lev2_path
        self.midi_file_name = midi_file_name
        self.is_fill = is_fill
        self.all_styles = all_styles
        self.type = type_
        self.also_found_at = also_found_at if isinstance(also_found_at, str) else '-'
        self.root_dir = root_dir
        self.exact_path = os.path.join(root_dir, lev1_path, lev2_path, midi_file_name) if exact_path is None else exact_path

    def __repr__(self):
        return f"<MidiMetadata(genre='{self.genre}', from_collection={self.collection}, midi_file_name='{self.midi_file_name}')>"

    @classmethod
    def from_dataframe_row(cls, row, root_dir):
        """Constructs a MidiMetadata object from a pandas Series (row)"""
        TimeSigKey = 'TimeSignatures' if 'TimeSignatures' in row else 'time_signatures'
        
        if not row[TimeSigKey].startswith("["):
            row[TimeSigKey] = "[" + row[TimeSigKey]

        if not row[TimeSigKey].endswith("]"):
            row[TimeSigKey] = row[TimeSigKey] + "]"

        return cls(
            length_group=row['LengthGroup'] if 'LengthGroup' in row else row['length_group'],
            num_measures=row['NumMEASURES'] if 'NumMEASURES' in row else row['num_measures'],
            collection=row['Collection'] if 'Collection' in row else row['collection'],
            time_signatures=row[TimeSigKey],
            tempo=row['Tempo'] if 'Tempo' in row else row['tempo'],
            is_performed=row['IsPerformed'] if 'IsPerformed' in row else row['is_performed'],
            genre=row['Genre'] if 'Genre' in row else row['genre'],
            lev1_path=row['Lev1Path'] if 'Lev1Path' in row else row['lev1_path'],
            lev2_path=row['Lev2Path'] if 'Lev2Path' in row else row['lev2_path'],
            midi_file_name=row['MidiFileName'] if 'MidiFileName' in row else row['midi_file_name'],
            is_fill=row['IsFill'] if 'IsFill' in row else row['is_fill'],
            all_styles=row['AllStyles'] if 'AllStyles' in row else row['all_styles'],
            type_=row['Type'] if 'Type' in row else row['type'],
            also_found_at=row['AlsoFoundAt'] if 'AlsoFoundAt' in row else row['also_found_at'],
            root_dir=root_dir,
            exact_path=row['exact_path'] if 'exact_path' in row else None
        )

    def note_sequence(self):
        """Return a NoteSequence object from the MIDI file"""
        # try to load the file from different levels of the directory
        path = self.exact_path
        num_dirs = len(path.split('/'))
        while not os.path.exists(path) and num_dirs > 0:
            path = '/'.join(path.split('/')[1:])
            num_dirs -= 1
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at any level of : {self.exact_path}")
        return note_seq.midi_file_to_note_sequence(path)

    def hvo_sequence(self, drum_mapping=None):
        """Return a HVO_Sequence object from the MIDI file"""
        if not _HAS_HVO_SEQ:
            raise ImportError("hvo_sequence is not installed. Please install it using `pip install hvo_sequence`")
        if drum_mapping is None:
            drum_mapping = ROLAND_REDUCED_MAPPING
        hvo_seq = midi_to_hvo_sequence(self.exact_path, drum_mapping=drum_mapping, beat_division_factors=[4])
        hvo_seq.metadata = self.__dict__
        return hvo_seq

    def split_into_2bar_segs(self):
        """Split the MIDI file into 2-bar segments and return a list of NoteSequence objects"""

        ns_ = self.note_sequence()

        # Calculate total steps in the sequence
        # Assuming a constant tempo for simplicity
        tempo = ns_.tempos[0].qpm if len(ns_.tempos) > 0 else 120.0  # Defaulting to 120 BPM if not provided
        bar_duration = 60.0 / tempo * 4

        chunks = []

        if len(ns_.tempos) == 0 or len(ns_.time_signatures) == 0:
            return chunks

        start_time = 0
        count = 0
        while not (ns_.total_time - start_time < bar_duration):
            # Calculate start and end step
            start_time = count * bar_duration
            end_time = min(start_time + bar_duration * 2, ns_.total_time)

            # Extract subsequence
            note_seq_ = note_seq.NoteSequence()

            note_seq_.tempos.add(qpm=ns_.tempos[0].qpm, time=0)
            note_seq_.time_signatures.add(numerator=ns_.time_signatures[0].numerator,
                                          denominator=ns_.time_signatures[0].denominator,
                                          time=0)

            # Adjust timings of the extracted notes
            for note in ns_.notes:
                if end_time > note.start_time > start_time:
                    note_seq_.notes.add(pitch=note.pitch,
                                        velocity=note.velocity,
                                        start_time=note.start_time - start_time,
                                        end_time=min(note.end_time - start_time, end_time - start_time),
                                        is_drum=note.is_drum)

            # Copy global attributes
            chunks.append(note_seq_)

            count += 1

        return chunks



class MidiCollection:
    def __init__(self, midi_metadata_list=None):
        self.midi_files = midi_metadata_list if midi_metadata_list else []

    def _load_from_csv(self, csv_path):
        """Load MIDI metadata from a CSV file and populate the midi_files list"""
        root_dir = os.path.dirname(csv_path)
        self.root_dir = root_dir
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            midi_metadata = MidiMetadata.from_dataframe_row(row, root_dir)
            self.midi_files.append(midi_metadata)

    @classmethod
    def from_csv(cls, csv_path):
        """Class method to create a MidiCollection instance from a CSV file"""
        instance = cls()
        instance._load_from_csv(csv_path)
        return instance

    def get_dataframe(self):
        """Return a pandas DataFrame containing the MIDI metadata"""
        return pd.DataFrame([midi.__dict__ for midi in self.midi_files])

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, index):
        return self.midi_files[index]

    def __repr__(self):
        return f"<MidiCollection(size={len(self)})>"

    def filter_by_length(self, length_group):
        """Return a MidiCollection object filtered by the specified length group (short, medium, or long)"""
        if length_group not in ['short', 'medium', 'long']:
            raise ValueError("length_group must be one of ['short', 'medium', 'long']")
        filtered_midi_files = [midi for midi in self.midi_files if midi.length_group == length_group]
        return MidiCollection(filtered_midi_files)

    def get_short_midi_files(self):
        """Return a subset of MidiMetadata objects that are short"""
        return self.filter_by_length('short')

    def get_medium_midi_files(self):
        """Return a subset of MidiMetadata objects that are medium"""
        return self.filter_by_length('medium')

    def get_long_midi_files(self):
        """Return a subset of MidiMetadata objects that are long"""
        return self.filter_by_length('long')

    def with_genre_tags(self):
        """Return a MidiCollection object containing only the MIDI files that have genre tags"""
        filtered_midi_files = [midi for midi in self.midi_files if midi.all_styles and midi.all_styles != '-']
        return MidiCollection(filtered_midi_files)

    def without_genre_tags(self):
        """Return a MidiCollection object containing only the MIDI files that don't have genre tags"""
        filtered_midi_files = [midi for midi in self.midi_files if not midi.all_styles or midi.all_styles == '-']
        return MidiCollection(filtered_midi_files)

    def genre_distribution(self):
        """Return the distribution of unique genre groups in the MIDI collection"""

        def extract_genres(genre_str):
            """Extract and sort genres from the genre string"""
            genres = genre_str.split('+')
            return tuple(sorted(genres))

        # Create a defaultdict to store counts of each unique genre group
        genre_counts = defaultdict(int)

        # Count genre distributions per unique group
        for midi in self.midi_files:
            genre_str = midi.all_styles
            if genre_str:  # Check if the genre string is not None
                unique_genres = extract_genres(genre_str)
                genre_counts[unique_genres] += 1

        # Convert defaultdict to regular dictionary and sort by counts
        sorted_genre_counts = dict(sorted(genre_counts.items(), key=lambda item: item[1], reverse=True))

        return sorted_genre_counts


    def filter_by_major_genres(self):
        """Filter the collection by major genres. Exclude files that match more than one major genre."""

        major_genres = ["Rock", "Pop", "Jazz", "Blues", "Funk", "Latin", "Afrobeat",
                        "Reggae", "Country", "Hiphop", "Disco", "Soul", "Electronic"]

        # Function to count how many major genres a midi file belongs to
        def count_genre_matches(genre_str):
            return sum(1 for major_genre in major_genres if major_genre.lower() in genre_str.lower())

        filtered_midi_files = [midi for midi in self.midi_files
                               if count_genre_matches(midi.all_styles) == 1]

        return MidiCollection(filtered_midi_files)

    def filter_by_source_set(self, source_set):
        """Filter the collection by source set"""
        filtered_midi_files = [midi for midi in self.midi_files if midi.collection == source_set]
        return MidiCollection(filtered_midi_files)

    def get_all_sources(self):
        """Return a list of all unique sources in the collection"""
        return list(set(midi.collection for midi in self.midi_files))

    def remove_genres(self, genres):
        if isinstance(genres, str):
            genres = [genres]

        filtered_midi_files = [midi for midi in self.midi_files
                                 if not any(genre.lower() in midi.all_styles.lower() for genre in genres)]
        return MidiCollection(filtered_midi_files)

    def with_meter_tags(self):
        """Return a MidiCollection object containing only the MIDI files that have meter tags"""
        filtered_midi_files = [midi for midi in self.midi_files if midi.time_signatures and midi.time_signatures != '[]']
        return MidiCollection(filtered_midi_files)

    def without_meter_tags(self):
        """Return a MidiCollection object containing only the MIDI files that don't have meter tags"""
        filtered_midi_files = [midi for midi in self.midi_files if
                               not midi.time_signatures or midi.time_signatures == '[]']
        return MidiCollection(filtered_midi_files)

    def with_tempo_metadata(self):
        """Return a MidiCollection object containing only the MIDI files that have tempo metadata"""
        filtered_midi_files = [midi for midi in self.midi_files if midi.tempo and midi.tempo != '[]']
        return MidiCollection(filtered_midi_files)

    def without_tempo_metadata(self):
        """Return a MidiCollection object containing only the MIDI files that don't have tempo metadata"""
        filtered_midi_files = [midi for midi in self.midi_files if not midi.tempo or midi.tempo == '[]']
        return MidiCollection(filtered_midi_files)

    def meter_distribution(self):
        """Return the distribution of unique meter groups in the MIDI collection"""

        def extract_meters(meter_str):
            """Extract and sort meters from the meter string"""
            meters = meter_str.split('&')
            return ' & '.join(sorted(meters))

        # Create a defaultdict to store counts of each unique meter group
        meter_counts = defaultdict(int)

        # Count meter distributions per unique group
        for midi in self.midi_files:
            meter_str = midi.time_signatures
            if meter_str:  # Check if the meter string is not None
                unique_meters = extract_meters(meter_str)
                meter_counts[unique_meters] += 1

        # Convert defaultdict to regular dictionary and sort by counts
        sorted_meter_counts = dict(sorted(meter_counts.items(), key=lambda item: item[1], reverse=True))

        return sorted_meter_counts

    def tempo_distribution(self):
        """Return basic statistics for the tempo values in the MIDI collection"""

        # Extract and parse tempo values
        tempos = []
        for midi in self.midi_files:
            if midi.tempo and midi.tempo.startswith("[") and midi.tempo.endswith("]"):
                tempo_values = [float(t.strip()) for t in midi.tempo[1:-1].split(",")]
                tempos.extend(tempo_values) if tempo_values else None

        if tempos and len(tempos) > 0:
            # Compute statistics
            min_tempo = min(tempos)
            max_tempo = max(tempos)
            mean_tempo = sum(tempos) / len(tempos)
            median_tempo = sorted(tempos)[len(tempos) // 2]
            variance_tempo = sum((t - mean_tempo) ** 2 for t in tempos) / len(tempos)
            std_dev_tempo = variance_tempo ** 0.5

            return {
                'min': min_tempo,
                'max': max_tempo,
                'mean': mean_tempo,
                'median': median_tempo,
                'std_dev': std_dev_tempo
            }
        else:
            return {'min': None, 'max': None, 'mean': None, 'median': None, 'std_dev': None}

    def filter_unique_midis(self):
        """
        Filters the collection to retain only unique MIDI files for each group.
        Discards any short MIDI file also found in medium or long groups.
        Discards any medium MIDI file also found in long group.
        """

        unique_midi_files = []

        for midi_ in self.midi_files:
            if midi_.length_group == 'short':
                if 'medium' in midi_.also_found_at or 'long' in midi_.also_found_at:
                    continue
                else:
                    unique_midi_files.append(midi_)
            elif midi_.length_group == 'medium':
                if 'long' in midi_.also_found_at:
                    continue
                else:
                    unique_midi_files.append(midi_)
            elif midi_.length_group == 'long':
                unique_midi_files.append(midi_)

        return MidiCollection(unique_midi_files)

    def get_4_4_midis(self):
        """Return a MidiCollection object containing only the MIDI files that have 4/4 time signature"""
        filtered_midi_files = [midi for midi in self.midi_files if
                               ',' not in midi.time_signatures and '4_4' in midi.time_signatures]

        return MidiCollection(filtered_midi_files)

    def get_performed_midis(self):
        """Return a MidiCollection object containing only the MIDI files that have a performer"""
        filtered_midi_files = [midi for midi in self.midi_files if midi.is_performed == 1]
        return MidiCollection(filtered_midi_files)

    def get_programmed_midis(self):
        """Return a MidiCollection object containing only the MIDI files that have a performer"""
        filtered_midi_files = [midi for midi in self.midi_files if midi.is_performed == 0]
        return MidiCollection(filtered_midi_files)

    def filter_by_genre(self, genre):
        """Return a MidiCollection object containing only the MIDI files that match the specified genre"""
        filtered_midi_files = [midi for midi in self.midi_files if genre.lower() == midi.genre.lower()]
        return MidiCollection(filtered_midi_files)

    def filter_by_excluding_genre(self, genre):
        """Return a MidiCollection object containing only the MIDI files that don't match the specified genre"""
        filtered_midi_files = [midi for midi in self.midi_files if genre.lower() != midi.genre.lower()]
        return MidiCollection(filtered_midi_files)

    def get_n_random_midis(self, n):
        """Return a MidiCollection object containing n random MIDI files"""
        if n > len(self.midi_files):
            raise ValueError(f"n cannot be greater than the number of MIDI files in the collection ({len(self.midi_files)})")

        return MidiCollection(random.sample(self.midi_files, n))

    def get_n_samples_per_genre(self, target_num_samples_per_genre):
        selected_midis = []


        for genre in self.get_unique_genres():
            # get data in column genre
            genre_subset = self.filter_by_genre(genre)
            groove_midi_samples = genre_subset.filter_by_source_set("GrooveMIDI")
            non_groove_midi_samples = genre_subset.filter_by_excluding_genre("GrooveMIDI")

            # if groove midi samples are larger than target, take random samples from groove midi
            if len(groove_midi_samples) >= target_num_samples_per_genre:
                selected_midis.extend(groove_midi_samples.get_n_random_midis(target_num_samples_per_genre))
            elif len(genre_subset) >= target_num_samples_per_genre:
                # if groove midi samples are less than target, take all groove midi samples and take random samples from non-groove midi
                selected_midis.extend(groove_midi_samples)
                selected_midis.extend(
                    non_groove_midi_samples.get_n_random_midis(target_num_samples_per_genre - len(groove_midi_samples)))
            else:
                # if genre samples are less than target, take all groove midi samples and all non-groove midi samples
                selected_midis.extend(groove_midi_samples.midi_files)
                selected_midis.extend(non_groove_midi_samples.midi_files)
                remaining_samples = target_num_samples_per_genre - len(groove_midi_samples) - len(
                    non_groove_midi_samples)
                remaining_samples_to_total = np.ceil(remaining_samples / len(genre_subset))
                remaining_sample_midis = []
                for i in range(int(remaining_samples_to_total)):
                    random.shuffle(genre_subset.midi_files)
                    remaining_sample_midis.extend(genre_subset)
                remaining_sample_midis = remaining_sample_midis[:remaining_samples]
                selected_midis.extend(remaining_sample_midis)

        return MidiCollection(selected_midis)

    def get_source_genre_distribution(self):
        dataframe = pd.DataFrame()
        # add source names as column headers
        sources = self.get_all_sources()

        genres_per_source = {}

        for source in sources:
            genres_per_source[source] = self.filter_by_source_set(source).genre_distribution()

        # use source names as rows and genres as columns
        dataframe = pd.DataFrame(genres_per_source).T
        dataframe = dataframe.fillna(0)

        # add a row and column for the total number of genres
        dataframe['Total'] = dataframe.sum(axis=1)
        dataframe.loc['Total'] = dataframe.sum(axis=0)

        return dataframe

    def dump(self, dir_path):
        """Dump the collection to a directory"""

        os.makedirs(dir_path, exist_ok=True)

        readme = f"""
        # MIDI Collection

        ## Summary
        This collection contains {len(self.midi_files)} MIDI files.

        This subset has no genre tags, but all files have 4/4 time signature.    

        ## Performed MIDI files
        Performed Files: {len(self.get_performed_midis().midi_files)}
        Mechanical Files: {len(self.get_programmed_midis().midi_files)}

        ## Genre Distribution
        {self.genre_distribution()}

        ## Meter Distribution
        {self.meter_distribution()}

        ## Tempo Distribution
        {self.tempo_distribution()}

        """

        new_paths = []

        for midi in self.midi_files:
            existing_root = midi.root_dir
            current_path = midi.exact_path
            new_path = current_path.replace(existing_root, dir_path)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy(current_path, new_path)
            new_paths.append(new_path)

        if readme:
            with open(os.path.join(dir_path, 'README.md'), 'w') as f:
                f.write(readme)

        df = self.get_dataframe()
        df['exact_path'] = new_paths

        # save the summary csv
        df.to_csv(os.path.join(dir_path, 'summary.csv'), index=False)

        print(f'Dumped {len(self.midi_files)} MIDI files to {dir_path}')

    def remap_genres_to_major(self):
        """Remap the genres using style mapping"""
        df = self.get_dataframe()

        # convert to list
        styles = df['all_styles'].tolist()

        # map the styles
        mapped_styles = [style_mapping[style] if style in style_mapping else 'Other' for style in styles ]

        for ix, style in enumerate(mapped_styles):
            self.midi_files[ix].genre = style
            self.midi_files[ix].all_styles = style

        df['all_styles'] = mapped_styles
        df['genre'] = mapped_styles

        # save the summary csv
        df.to_csv(os.path.join(self.root_dir, 'summary_remapped_genres.csv'), index=False)

        print(f"Remapped genres and saved to {self.root_dir}/summary_remapped_genres.csv")

    def get_unique_genres(self):
        """Return a list of unique genres in the collection"""
        return list(set(midi.all_styles for midi in self.midi_files))

    def split_into_train_test_validation(self, train_size, test_size, validation_size):
        """Split the collection into train, test, and validation sets"""
        assert train_size + test_size + validation_size == 1.0, "The sum of train_size, test_size, and validation_size must be 1.0"
        train_subset = []
        test_subset = []
        validation_subset = []

        for genre in self.get_unique_genres():
            genre_subset = self.filter_by_genre(genre).midi_files
            # shuffle the genre subset
            random.shuffle(genre_subset)
            size_of_genre = len(genre_subset)
            train_sample_size = int(train_size * size_of_genre)
            test_sample_size = int(test_size * size_of_genre)
            validation_sample_size = int(validation_size * size_of_genre)
            train_subset.extend(genre_subset[:train_sample_size])
            test_subset.extend(genre_subset[train_sample_size:(train_sample_size + test_sample_size)])
            validation_subset.extend(genre_subset[(train_sample_size + test_sample_size):])

        return MidiCollection(train_subset), MidiCollection(test_subset), MidiCollection(validation_subset)

class CollectionModifier:
    def __init__(self, collection: MidiCollection):
        self.df = collection.get_dataframe()
        self.collection = collection

    def rename_midi_files(self, save_dir):
        """Rename the MIDI files in the collection and save them to a directory"""

        # Create the root_identifier column
        self.df["root_identifier"] = self.df["lev1_path"] + "-" + self.df["lev2_path"]

        # Create the mapping for root_identifier
        unique_root_identifiers = self.df["root_identifier"].unique()
        map = {root_identifier: "p" + str(i + 1).zfill(10) for i, root_identifier in enumerate(unique_root_identifiers)}

        # Map the root_identifiers
        self.df['root_identifier'] = self.df['root_identifier'].map(map)

        # Generate the renamed_midi_file_name column
        # Using cumcount to get a cumulative count within each group
        self.df['order_in_group'] = self.df.groupby(['lev2_path']).cumcount() + 1
        fill_type = self.df['is_fill'].map({False: 'beat', True: 'fill'})

        self.df['renamed_midi_file_name'] = (
                self.df['collection'] + "__" + self.df['length_group'] + '/' + self.df['root_identifier'] + '__m' + self.df['order_in_group'].astype(str).str.zfill(3) +
                '__' + fill_type + '__'+ self.df.all_styles +'.mid')

        # Drop the temporary 'order_in_group' column
        self.df.drop(columns=['order_in_group'], inplace=True)

        """Dump the collection to a directory"""
        os.makedirs(save_dir, exist_ok=True)

        old_paths = self.df['exact_path'].tolist()

        if save_dir.endswith('/'):
            save_dir = save_dir[:-1]

        if 'root_identifier' in self.df and 'renamed_midi_file_name' in self.df:
            new_paths = save_dir + '/midis/' + self.df['renamed_midi_file_name']
        else:
            new_paths = self.df['exact_path'].replace(self.collection.root_dir, save_dir)

        for old_path, new_path in zip(old_paths, new_paths):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.copy(old_path, new_path)

        # swap the exact_path column with the new paths
        self.df['exact_path'] = new_paths

        self.df.to_csv(os.path.join(save_dir, 'summary.csv'), index=False)

        print(f'Dumped {len(self.df)} MIDI files to {save_dir}')
        print(f"New paths are in the {os.path.join(save_dir, 'summary.csv')} file")

    def split_in_2bar_segments(self, save_dir):
        """Split the MIDI files in the collection into 2-bar segments and save them to a directory
            skips those that are not 4-4 and have tempo changes

            Args:
                save_dir: the directory to save the new MIDI files
        """
        # Check and fix save_dir format
        if save_dir.endswith('/'):
            save_dir = save_dir[:-1]

        # Prepare for collecting new rows
        new_rows = []
        created_dirs = set()  # This will store directories we've already created

        # Use .at for faster access by label
        for index in tqdm.tqdm(self.df.index):
            midi_metadata = self.collection[index]
            chunks = midi_metadata.split_into_2bar_segs()
            for i, chunk in enumerate(chunks):
                new_row = self.df.loc[index].copy()
                appendix = f'__{str(i).zfill(3)}:{str(i + 2).zfill(3)}.mid'
                new_row['new_path'] = new_row['exact_path'].replace(
                    '.mid', appendix).replace('midis',
                                              'midis_2bar').replace(
                    self.collection.root_dir, save_dir)

                # Check and create directory only if it hasn't been created before
                dir_path = os.path.dirname(new_row['new_path'])
                if dir_path not in created_dirs:
                    os.makedirs(dir_path, exist_ok=True)
                    created_dirs.add(dir_path)

                note_seq.sequence_proto_to_midi_file(chunk, new_row['new_path'])
                new_rows.append(new_row)

        # Convert the list of new rows to a DataFrame
        df_temp = pd.DataFrame(new_rows)
        df_temp['exact_path'] = df_temp['new_path']
        df_temp.drop(columns=['new_path'], inplace=True)
        df_temp.to_csv(os.path.join(save_dir, 'summary.csv'), index=False)

        print(f'Dumped {len(df_temp)} MIDI files to {save_dir}')




if __name__ == '__main__':
    midi_collection = MidiCollection.from_csv('compiled/summary.csv')

    modifier = CollectionModifier(midi_collection)
    modifier.split_in_2bar_segments('test')
    #
    # modifier.rename_midi_files()
    #
    # # modifier.df.to_csv('test.csv', index=False)
    #
    # modifier.dump('test')
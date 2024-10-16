import miditok
import symusic
from miditoolkit import MidiFile

# Load a MIDI file
with open('2_bar.mid', 'rb') as file:
    midi_data = file.read()

# Initialize a tokenizer (let's use the REMI tokenizer for this example)
tokenizer = miditok.MIDILike()

# Tokenize the MIDI file
syscore = symusic.Score.from_midi(midi_data)

tokens = tokenizer.encode(symusic.Score.from_midi(midi_data))

# Optionally, you can print out the tokenized representation
for track_tokens in tokens:

    print(track_tokens)


track_tokens = tokens[0]
track_tokens.__dict__['tokens']
# General MIDI Level 1 PERCUSSION KEY MAP
# REF: https://www.midi.org/specifications-old/item/gm-level-1-sound-set
GM1_FULL_MAP = {
    "Acoustic_Bass_Drum": [35],
    "Bass_Drum_1": [36],
    "Side_Stick": [37],
    "Acoustic_Snare": [38],
    "Hand_Clap": [39],
    "Electric_Snare": [40],
    "Low_Floor_Tom": [41],
    "Closed_Hi_Hat": [42],
    "High_Floor_Tom": [43],
    "Pedal_Hi-Hat": [44],
    "Low_Tom": [45],
    "Open_Hi-Hat": [46],
    "Low-Mid_Tom": [47],
    "Hi-Mid_Tom": [48],
    "Crash_Cymbal_1": [49],
    "High_Tom": [50],
    "Ride_Cymbal_1": [51],
    "Chinese_Cymbal": [52],
    "Ride_Bell": [53],
    "Tambourine": [54],
    "Splash_Cymbal": [55],
    "Cowbell": [56],
    "Crash_Cymbal_2": [57],
    "Vibraslap": [58],
    "Ride_Cymbal_2": [59],
    "Hi_Bongo": [60],
    "Low_Bongo": [61],
    "Mute_Hi_Conga": [62],
    "Open_Hi_Conga": [63],
    "Low_Conga": [64],
    "High_Timbale": [65],
    "Low_Timbale": [66],
    "High_Agogo": [67],
    "Low_Agogo": [68],
    "Cabasa": [69],
    "Maracas": [70],
    "Short_Whistle": [71],
    "Long_Whistle": [72],
    "Short_Guiro": [73],
    "Long_Guiro": [74],
    "Claves": [75],
    "Hi_Wood_Block": [76],
    "Low_Wood_Block": [77],
    "Mute_Cuica": [78],
    "Open_Cuica": [79],
    "Mute_Triangle": [80],
    "Open_Triangle": [81]
}

# MAGENTA MAPPING
ROLAND_REDUCED_MAPPING = {
    "KICK": [36, 35],
    "SNARE": [38, 37, 40, 39],
    "HH_CLOSED": [42, 22, 44],
    "HH_OPEN": [46, 26],
    "TOM_3_LO": [43, 58, 41],
    "TOM_2_MID": [47, 45],
    "TOM_1_HI": [50, 48],
    "CRASH": [49, 52, 55, 57],
    "RIDE":  [51, 53, 59]
}

# GrooveToolbox 5 Part Kit Mappings
Groove_Toolbox_5Part_keymap = {
    "kick": [36, 35],
    "snare": [38, 37, 40],
    "closed": [42, 22, 44, 51, 53, 59],     # Closed cymbals (hihat and ride)
    "open": [46, 26, 49, 52, 55, 57],       # Open cymbals (open hihat, crash)
    "Toms": [43, 58, 47, 45, 50, 48, 41]    # Toms (low mid and high)
}

# GrooveToolbox 3 Part Kit Mappings
Groove_Toolbox_3Part_keymap = {
    "low": [36, 35],                                        # Kick
    "mid": [38, 37, 40, 43, 58, 47, 45, 50, 48, 41],        # Snare and Tom
    "hi": [42, 22, 44, 51, 53, 59, 46, 26, 49, 52, 55, 57]  # Hats/Crashes/Rides
}

# GrooveToolbox General MIDI Mappings
Groove_Toolbox_GM_keymap = {
    "kick": [35, 36],
    "snare": [37, 38, 40],
    "closed hihat": [42, 44],
    "open hihat": [46],
    "ride": [51, 53, 59],
    "crash": [49, 57],
    "extra cymbal": [52, 55],
    "low tom": [41, 43, 45],
    "mid tom": [47, 48],
    "high tom": [50]
}

# https://bernhard.hensler.net/roland-td-11-setup-midi-mapping/
# REF: https://rolandus.zendesk.com/hc/en-us/articles/360005173411-TD-17-Default-Factory-MIDI-Note-Map
ROLAND_TD_17_Full_map = {
    "KICK": [36],
    "SNARE_HEAD": [38],
    "SNARE_RIM": [40],
    "SNARE_X_Stick": [37],
    "TOM_1": [48],
    "TOM_1_RIM": [50],
    "TOM_2": [45],
    "TOM_2_RIM": [47],
    "TOM_3_HEAD": [43],
    "TOM_3_RIM": [58],
    "HH_OPEN_BOW": [46],
    "HH_OPEN_EDGE": [26],
    "HH_CLOSED_BOW": [42],
    "HH_CLOSED_EDGE": [22],
    "HH_PEDAL": [44],
    "CRASH_1_BOW": [49],
    "CRASH_1_EDGE": [55],
    "CRASH_2_BOW": [57],
    "CRASH_2_EDGE": [52],
    "RIDE_BOW": [51],
    "RIDE_EDGE": [59],
    "RIDE_BELL": [53]
}


def get_drum_mapping_using_label(drum_map_label):
    """
    returns the mapping dictionary for a given drum map using a string label identifier

    :param drum_map_label: (string matching the variable names above

    :return: a dictionary of drum maps
    """
    assert drum_map_label in ["GM1_FULL_MAP", "ROLAND_REDUCED_MAPPING", "Groove_Toolbox_5Part_keymap",
                              "Groove_Toolbox_3Part_keymap", "Groove_Toolbox_GM_keymap", "ROLAND_TD_17_Full_map"
                              ], "DRUM MAP LABEL IS INCORRECT"

    if drum_map_label == "GM1_FULL_MAP":
        return GM1_FULL_MAP
    elif drum_map_label == "ROLAND_REDUCED_MAPPING":
        return ROLAND_REDUCED_MAPPING
    elif drum_map_label == "Groove_Toolbox_5Part_keymap":
        return Groove_Toolbox_5Part_keymap
    elif drum_map_label == "Groove_Toolbox_3Part_keymap":
        return Groove_Toolbox_3Part_keymap
    elif drum_map_label == "Groove_Toolbox_GM_keymap":
        return Groove_Toolbox_GM_keymap
    elif drum_map_label == "ROLAND_TD_17_Full_map":
        return ROLAND_TD_17_Full_map



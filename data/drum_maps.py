CANONICAL_DRUM_MAP = {
    36: 'kick',
    38: 'snare',
    42: 'hh_closed',
    46: 'hh_open',
    43: 'low_tom',
    47: 'mid_tom',
    50: 'high_tom',
    49: 'crash',
    51: 'ride',
}


ROLAND_TD_11_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [36], 
    }, 
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [38, 37, 40], 
    }, 
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [42, 22, 44], 
    }, 
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [46, 26], 
    }, 
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [43, 58], 
    }, 
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [47, 45], 
    }, 
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [50, 48], 
    }, 
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [49, 52, 55, 57], 
    }, 
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [51, 53, 59]
    },
}

# General MIDI Reduced Drum Map (9-class)
GM_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [35, 36],  # Acoustic + Bass Drum 1
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [37, 38, 39, 40],  # Snare, clap, stick and variants
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [42, 44, 54, 82],  # Closed HH, Pedal HH, tambourine, shaker
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [46, 26],  # Open HH
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [41, 45, 61, 64, 66, 68],  # Low Floor Tom, Low Tom, Low Bongo
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [43, 47],  # Mid Tom, Low-Mid Tom
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [48, 50, 60, 62, 63, 65, 67],  # High Tom, High-Mid Tom, High Bongo
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [49, 52, 55, 57],  # Crash 1/2, Splash, Chinese Cymbal
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [51, 53, 56, 59, 76, 77],  # Ride 1/2, Ride Bell, Cowbell
    },
}

AUTUMN_50S_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [
            31, 32, 36, 60, 82, 111  # Kick and variants
        ]
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            101, 103, 38, 105, 107, 40, 39, 37, 62, 63, 64, 61, 81, 83, 84, 86, 33, 34, 35
        ]  # Snare, clap, stick and variants
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            99, 97, 66, 104, 102, 42, 109, 106, 68, 44  # Closed Tips, Pedals, Brushes, Tight
        ]
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            70, 76, 77, 78, 79, 80, 46, 85, 87, 90  # Open variations, pedal, brush
        ]
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            110, 108, 41, 43, 65, 67, 72, 74, 88, 89  # rack tom 3 and variants
        ]
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [
            113, 112, 45, 69, 73, 88, 93  # rack tom 2 and variants
        ]
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            117, 115, 47, 71, 75, 91, 95, 96, 27, 28  # rack tom 1, bongo and variants
        ]
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            58, 48, 49, 51, 57, 50, 59, 22, 25, 92, 55, 54, 56, 24, 94  # crash and variants
        ]
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [
            52, 53, 59, 57, 58, 23, 95, 96, 26, 29, 30, 37  # ride, triangle, wood block and variants
        ]
    },
}

EARLY_60S_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [
            36, 60, 82, 99  # Kick and variants
        ]
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            81, 83, 38, 84, 86, 40, 39, 37, 62, 63, 64, 61, 33, 34, 35, 101  # Snare, clap, stick and variants
        ]
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            87, 85, 66, 92, 90, 42, 97, 94, 68, 44, 31, 32  # hihat closed, tambourine and variants
        ]
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            70, 76, 77, 78, 79, 80, 46  # hihat open and variants
        ]
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            89, 88, 41, 43, 65, 72, 73, 91, 96  # Floor tom and variants
        ]
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': []
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            93, 95, 45, 47, 69, 74, 98  # rack tom and variants
        ]
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            49, 48, 50, 22, 54, 55, 53, 56, 24  # crash and variants
        ]
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [
            51, 53, 52, 23, 57, 59, 58, 25  # ride and variants
        ]
    },
}

OPEN_70S_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [
            36, 82, 60  # Kick and variants
        ]
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            81, 83, 38, 84, 86, 40, 39, 37, 62, 63, 64, 61, 33, 34, 35, 101 # Snare, clap, stick and variants
        ]
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            87, 85, 66, 92, 90, 42, 97, 94, 68, 44, 26, 32, 31  # hihat closed, shaker, tambourine and variants
        ]
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            70, 76, 77, 78, 79, 80, 46  # hihat open and variants
        ]
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            89, 93, 88, 91, 41, 43, 65, 67, 72, 73  # Floor tom low and variants
        ]
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [
            96, 95, 45, 69, 74  # floow tom hi and variants
        ]
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            100, 98, 47, 71, 75  # rack tom and variants
        ]
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            49, 48, 50, 22, 55, 54, 56, 23  # crash and variants
        ]
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [
            51, 53, 52, 24, 58, 59, 57, 25, 29, 30, 27, 28 # Ride, cowbell and variants
        ]
    },
}

BLACK_80S_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [36, 82, 60, 99],  # Kick and variants
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            81, 83, 38, 84, 86, 40, 39, 37, 62, 63, 64, 61, 101, 102, 35, 33, 34  # Snare, clap, stick and variants
        ],
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            87, 85, 66, 92, 90, 42, 97, 94, 68, 44, 31, 32  # hihat closed, tambourine and variants
        ],
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            70, 76, 77, 78, 79, 80, 46  # hihat open and variants
        ],
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            89, 93, 88, 91, 41, 43, 65, 67, 72, 73, 103, 104  # rack tom low, floor tom  and variants
        ],
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [
            96, 95, 45, 69, 74, 105  # rack tom 2 and variants
        ],
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            100, 98, 47, 71, 75, 106, 107  # rack tom hi and variants
        ],
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            49, 48, 50, 22, 55, 54, 56, 24, 57, 58, 25, 59, 26  # crash, splash, china and variants
        ],
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [
            51, 53, 52, 23, 21, 19, 17, 29, 30, 27, 28  # ride, cowbell, wood block and variants
        ],
    },
}

SPARKLE_MODERN_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [36, 82, 60],  # kick and variants
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            81, 83, 38, 84, 86, 40, 39, 37, 62, 63, 64, 61, 101, 102, 35, 33, 34  # Snare, clap, stick and variants
        ],
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            87, 85, 66, 92, 90, 42, 97, 94, 68, 44, 31, 32, 29, 27, 28  # hihat closed, tambourine, chopper and variants
        ],
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            70, 76, 77, 78, 79, 80, 46  # hihat open and variants
        ],
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            93, 91, 43, 67, 73, 89, 88, 41, 65, 72  # rack tom low, floor tom and variants
        ],
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [
            96, 95, 45, 69, 74  # rack tom mid and variants
        ],
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            100, 98, 47, 71, 75  # rack tom hi and variants
        ],
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            49, 48, 50, 22, 55, 54, 56, 24, 57, 58, 25, 59, 26  # crash, splash, china and variants
        ],
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [
            51, 53, 52, 23, 19, 21, 17, 16 # ride, cowbell, wood block and variants
        ],
    },
}


SESSION_STUDIO_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [36, 60],  # kick and variants
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            33, 34, 35, 37, 38, 39, 40, 61, 62, 63, 64, 81, 83, 84, 86  # Snare, clap, stick and variants
        ],
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            31, 32, 42, 44, 66, 68, 85, 87, 90, 92, 94, 97  # hihat closed, tambourine and variants
        ],
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            46, 70, 76, 77, 78, 79, 80  # hihat open and variants
        ],
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            41, 43, 65, 67, 72, 73, 88, 89, 93  # tom 3, tom 4 and variants
        ],
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [
            45, 69, 74, 95, 96  # tom 2 and variants
        ],
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            47, 71, 75, 98, 100  # tom 1 and variants
        ],
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            22, 24, 26, 27, 28, 48, 49, 50, 54, 55, 56, 57, 58, 59], # crash, splash, china and variants
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [23, 25, 29, 30, 51, 52, 53],  # ride, wood block and variants
    },
}

EBONY_VINTAGE_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [36, 82, 60],  # kick and variants
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [
            101, 103, 38, 105, 107, 40, 39, 37, 62, 63, 64, 61, 81, 83, 84, 86, 33, 34, 35, 27, 26  # Snare, clap, stick, sandpaper and variants
        ],
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [
            99, 97, 66, 104, 102, 42, 109, 106, 68, 44, 31, 32  # hihat closed, tambourine and variants
        ],
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [
            70, 76, 77, 78, 79, 80, 46, 85, 87, 90  # hihat open and variants
        ],
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [
            110, 108, 41, 65, 72, 88, 89, 43, 67, 73  # floor tom and variants
        ],
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [],
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [
            112, 113, 45, 47, 69, 71, 73, 75, 91, 93  # rack tom and variants
        ],
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [
            49, 58, 48, 57, 50, 59, 22, 25, 92, 55, 54, 56, 24, 94  # crash and variants
        ],
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [
            51, 53, 52, 23, 95, 96, 28, 29, 30  # ride, cowbell and variants
        ],
    },
}

GM_EXTENDED_REDUCED_DRUM_MAP = {
    36: {
        'name': 'kick',
        'index': 0,
        'midi_notes': [34, 35, 36],  # kick and variants
    },
    38: {
        'name': 'snare',
        'index': 1,
        'midi_notes': [6, 33, 37, 38, 39, 40, 66, 67, 68, 69, 70, 71, 125, 126, 127],
    },
    42: {
        'name': 'hh_closed',
        'index': 2,
        'midi_notes': [7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 42, 61, 62, 63, 64, 65, 122],
    },
    46: {
        'name': 'hh_open',
        'index': 3,
        'midi_notes': [12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 44, 46, 60, 119, 120, 121, 123, 124],
    },
    43: {
        'name': 'low_tom',
        'index': 4,
        'midi_notes': [72, 73, 74, 75],
    },
    47: {
        'name': 'mid_tom',
        'index': 5,
        'midi_notes': [2, 3, 45, 47, 77, 78, 79, 80],
    },
    50: {
        'name': 'high_tom',
        'index': 6,
        'midi_notes': [1, 4, 5, 41, 43, 48, 50, 81, 82],
    },
    49: {
        'name': 'crash',
        'index': 7,
        'midi_notes': [27, 28, 29, 30, 31, 32, 49, 83, 94, 95, 106, 107],
    },
    51: {
        'name': 'ride',
        'index': 8,
        'midi_notes': [0, 51, 52, 53, 54, 55, 56, 57, 58, 59, 76, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118],
    },
}


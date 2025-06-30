'''
This script is used to get and serialize the GrooveMIDI dataset from the TensorFlow Datasets.

The dataset is available at https://www.tensorflow.org/datasets/catalog/groove

The dataset is split into 3 parts:
- full-midionly: full MIDI files
- 2bar-midionly: 2-bar MIDI files
- 4bar-midionly: 4-bar MIDI files

The dataset is split into 3 parts:
- full-midionly: full MIDI files
- 2bar-midionly: 2-bar MIDI files
- 4bar-midionly: 4-bar MIDI files
'''
import tensorflow_datasets as tfds
from symusic import Score
import os
import pickle


def process_feature(feature: tfds.features.FeaturesDict, drummer_label: tfds.features.ClassLabel, primary_style_label: tfds.features.ClassLabel, time_signature_label: tfds.features.ClassLabel, type_label: tfds.features.ClassLabel) -> dict:
    '''
    Process a single feature from the dataset and return a dictionary with the processed data.
    
    A feature is in the following format:
        
        FeaturesDict({
            'bpm': int32,
            'drummer': ClassLabel(shape=(), dtype=int64, num_classes=10),
            'id': string,
            'midi': string,
            'style': FeaturesDict({
                'primary': ClassLabel(shape=(), dtype=int64, num_classes=18),
                'secondary': string,
            }),
            'time_signature': ClassLabel(shape=(), dtype=int64, num_classes=5),
            'type': ClassLabel(shape=(), dtype=int64, num_classes=2),
        })
    
    Args:
        feature: A single feature from the dataset.
        drummer_label: The label for the drummer.
        primary_style_label: The label for the primary style.
        time_signature_label: The label for the time signature.
        type_label: The label for the type.
    Returns:
        A dictionary with the processed data.
    '''
    try:
        bpm = int(feature['bpm'])
        drummer_idx = int(feature['drummer'])
        id = feature['id'].decode()
        midi_bytes = feature["midi"]
        style_primary_idx = int(feature['style']['primary'])
        style_secondary = str(feature['style']['secondary'])
        time_signature_idx = int(feature['time_signature'])
        type_idx = int(feature['type'])

        # Convert indices to string labels
        drummer_str = drummer_label.int2str(drummer_idx)
        style_primary_str = primary_style_label.int2str(style_primary_idx)
        time_signature_str = time_signature_label.int2str(time_signature_idx)
        type_str = type_label.int2str(type_idx)

        entry = {
            "bpm": bpm,
            "drummer": drummer_str,
            "id": id,
            "style_primary": style_primary_str,
            "style_secondary": style_secondary,
            "time_signature": time_signature_str,
            "type": type_str,
            "midi": midi_bytes,
        }
        return entry
    except Exception as e:
        print(f"Error processing entry: {e}")
        return None


def process_dataset(type_name) -> dict:
    '''
    Process a dataset and return a dictionary of processed features for each split.

    Args:
        type_name: The name of the dataset to process.
    Returns:
        A dictionary of picklable processed features for each split.
    '''
    print(f"Processing {type_name}...")
    builder = tfds.builder(type_name)
    builder.download_and_prepare()
    info = builder.info

    # Get the ClassLabel objects
    drummer_label = info.features['drummer']
    primary_style_label = info.features['style']['primary']
    time_signature_label = info.features['time_signature']
    type_label = info.features['type']

    dataset_dict = {}

    for split_name in info.splits:
        print(f"Processing {split_name}...")
        dataset_dict[split_name] = []
        dataset = tfds.load(
            name=type_name,
            split=split_name,
            try_gcs=True
        )

        for feature in tfds.as_numpy(dataset):
            entry = process_feature(feature, drummer_label, primary_style_label, time_signature_label, type_label)
            if entry is None:
                exit(-1)
            dataset_dict[split_name].append(entry)

    return dataset_dict

if __name__ == "__main__":
    
    TYPES = ["groove/full-midionly", "groove/2bar-midionly", "groove/4bar-midionly"]

    # Create directory for the dataset
    path_of_script = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(path_of_script, "raw/GMD")
    os.makedirs(dataset_root, exist_ok=True)

    for type_name in TYPES:
        dataset_dict = process_dataset(type_name)
        out_name = os.path.join(dataset_root, f"{type_name.split('/')[-1]}.pkl")
        with open(out_name, "wb") as f:
            pickle.dump(dataset_dict, f)
        print(f"Saved {len(dataset_dict)} entries to {out_name}")
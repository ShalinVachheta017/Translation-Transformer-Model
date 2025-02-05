from pathlib import Path


def get_config():
    """
    Returns a dictionary with the default configuration for the translation
    transformer model training.

    The configuration dictionary contains the following keys:

    - batch_size: The batch size for training.
    - num_epochs: The number of epochs for training.
    - lr: The learning rate for the optimizer.
    - seq_len: The sequence length for the transformer model.
    - d_model: The embedding dimension for the transformer model.
    - datasource: The name of the dataset to use.
    - lang_src: The source language for the translation task.
    - lang_tgt: The target language for the translation task.
    - model_folder: The folder to save the model weights in.
    - model_basename: The base name for the model weights files.
    - preload: The preloading strategy for the model weights.
      Can be one of 'latest', 'best', or 'none'.
    - tokenizer_file: The filename format for the tokenizers.
    - experiment_name: The name of the experiment for the W&B run.

    Returns:
        dict: The configuration dictionary.
    """

    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder


def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

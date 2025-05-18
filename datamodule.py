import torch
import shutil
import tarfile
import warnings
import subprocess
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

def build_vocab(sequences):
    """
    Builds a vocabulary from the given sequences.
    Args:
        sequences (list): A list of sequences (words) to build the vocabulary from.
    Returns:
        dict: A dictionary mapping characters to indices.
    """
    # Initialize the vocabulary with special tokens
    # <pad> for padding, <sos> for start of sequence, and <eos> for end of sequence
    # <pad> is 0, <sos> is 1, and <eos> is 2
    # The rest of the characters will be indexed starting from 3
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    idx = 3
    for seq in sequences:
        for c in str(seq):
            if c not in vocab:
                vocab[c] = idx
                idx += 1
    return vocab

def tokenize(word, vocab):
    """
    Tokenizes a word into indices based on the provided vocabulary.
    Args:
        word (str): The word to tokenize.
        vocab (dict): The vocabulary mapping characters to indices.
    Returns:    
        list: A list of indices representing the tokenized word.
    """
    # Convert the word to a string and tokenize it using the vocabulary
    # Add <sos> and <eos> tokens
    return [vocab["<sos>"]] + [vocab[c] for c in str(word)] + [vocab["<eos>"]]

def dakshina_collate_fn(batch):
    """
    Custom collate function for the Dakshina dataset.
    Args:
        batch (list): A list of tuples containing native words, romanizations, and attestations.
    Returns:
        tuple: A tuple containing padded native words, padded romanizations, and stacked attestations.
    """
    native_words, romanizations, attestations = zip(*batch)
    native_words_padded = pad_sequence(native_words, batch_first=True, padding_value=0)
    romanizations_padded = pad_sequence(romanizations, batch_first=True, padding_value=0)
    attestations = torch.stack(attestations)
    return native_words_padded, romanizations_padded, attestations

class DakshinaDataset(Dataset):
    """
    A PyTorch Dataset for the Dakshina dataset.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing the dataset.
        native_vocab (dict): Vocabulary for native script.
        roman_vocab (dict): Vocabulary for romanization.
    """
    def __init__(self, dataframe, native_vocab, roman_vocab):
        self.data = dataframe
        self.native_vocab = native_vocab
        self.roman_vocab = roman_vocab

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the native script word, romanization, and attestation.
        """
        row = self.data.iloc[idx]
        native_word, romanization, attestation = row.iloc[0], row.iloc[1], row.iloc[2]

        native_tensor = torch.LongTensor(tokenize(native_word, self.native_vocab))
        roman_tensor = torch.LongTensor(tokenize(romanization, self.roman_vocab))
        attestation_tensor = torch.tensor(float(attestation), dtype=torch.float)
        return native_tensor, roman_tensor, attestation_tensor

class DakshinaDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Dakshina dataset.

    Args:
        data_dir (Path): Path to the dataset directory.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        lang (str): Language code for the dataset.
    """
    def __init__(self, data_dir=Path("./data/dakshina_dataset_v1.0"), batch_size=128, num_workers=1, lang="ta"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Validate the language code
        if lang not in ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "sd", "si", "ta", "te", "ur"]:
            raise ValueError("Invalid language code. Choose from ['bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur']")
        self.lang = lang

    def prepare_data(self):
        """
        Prepares the dataset by downloading and extracting it if not already available.
        """
        if self.data_dir.exists():
            print("Data directory already exists, skipping download.")
            return
        
        # URL for the dataset
        url = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"
        tar_path = Path(str(self.data_dir)+".tar")

        if tar_path.exists():
            print("Tar file already exists, skipping download.")
        else:
            print("Downloading dataset...")
            # Download the dataset using curl
            subprocess.run(["curl", "-o", str(tar_path), "-L", url], check=True)
        
        print("Extracting dataset...")
        # Extract the tar file
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=self.data_dir, filter="data")
            print("Extraction complete.")

        # Handle nested directory structure
        extracted_items = list(self.data_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            root_folder = extracted_items[0]
            print(f"Root folder detected: {root_folder.name}. Moving its contents...")
            for item in root_folder.iterdir():
                item.rename(self.data_dir / item.name)
            root_folder.rmdir()
            print("Root folder removed, contents moved to extract_path.")

        # Clean up unnecessary files and directories
        for dir_path in self.data_dir.iterdir():
            if dir_path.is_dir() and len(dir_path.name) == 2:
                lexicons_path = dir_path / "lexicons"
                if lexicons_path.exists() and lexicons_path.is_dir():                        
                    # Remove all contents in the 2-char directory except "lexicons"
                    for item in dir_path.iterdir():
                        if item != lexicons_path:
                            shutil.rmtree(item) if item.is_dir() else item.unlink()
                            
                    # Move files from "lexicons" to the parent directory
                    for item in lexicons_path.iterdir():
                        item.rename(dir_path / item.name)
                    
                    # Remove the "lexicons" folder
                    lexicons_path.rmdir()
                    print(f"Processed and cleaned directory: {dir_path.name}")
    
    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup (e.g., 'fit', 'test').
        """
        lang_dir = self.data_dir / self.lang
        train_data = lang_dir / f"{self.lang}.translit.sampled.train.tsv"
        val_data = lang_dir / f"{self.lang}.translit.sampled.dev.tsv"
        test_data = lang_dir / f"{self.lang}.translit.sampled.test.tsv"
        
        # Load the datasets into pandas DataFrames
        train_df = pd.read_csv(train_data, sep="\t", names=[self.lang, "en", "attest"])
        val_df = pd.read_csv(val_data, sep="\t", names=[self.lang, "en", "attest"])
        test_df = pd.read_csv(test_data, sep="\t", names=[self.lang, "en", "attest"])

        # Build vocabularies for native and romanized scripts
        self.native_vocab = build_vocab(train_df[self.lang])
        self.roman_vocab = build_vocab(train_df["en"])
        
        # Store the datasets as attributes for later use
        self.train_dataset = DakshinaDataset(train_df, self.native_vocab, self.roman_vocab)
        self.val_dataset = DakshinaDataset(val_df, self.native_vocab, self.roman_vocab)
        self.test_dataset = DakshinaDataset(test_df, self.native_vocab, self.roman_vocab)

    def train_dataloader(self):
        """
        Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=dakshina_collate_fn
        )

    def val_dataloader(self):
        """
        Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=dakshina_collate_fn
        )

    def test_dataloader(self):
        """
        Returns the DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=dakshina_collate_fn
        )


if __name__ == "__main__":
    # Instantiate the data module
    data_module = DakshinaDataModule()
    # Prepare the data (download and extract if necessary)
    data_module.prepare_data()
    # Set up the datasets
    data_module.setup()
    # Get the data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Print dataset and loader details
    print(f"Language: {data_module.lang}")
    print(f"Roman (Input) vocab size: {len(data_module.roman_vocab)}")
    print(f"Native (Output) vocab size: {len(data_module.native_vocab)}")
    print(f"Batch size: {data_module.batch_size}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
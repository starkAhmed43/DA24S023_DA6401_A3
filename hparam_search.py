import wandb
import argparse
from model import Seq2SeqModel
import pytorch_lightning as pl
from datamodule import DakshinaDataModule
from pytorch_lightning.loggers.wandb import WandbLogger

pl.seed_everything(42, workers=True)

# Main function to train the model
def hparam_search():
    # Initialize Wandb
    wandb.init(project='DA24S023_DA6401_A3', config=None)

    # Get hyperparameters from the Wandb configuration
    config = wandb.config

    # Instantiate DataModule for the chosen language (For example, 'hi' for Hindi)
    data_module = DakshinaDataModule(lang=config.lang, batch_size=config.batch_size, num_workers=config.num_workers)
    data_module.prepare_data()
    data_module.setup()

    # Instantiate the model
    input_vocab_size = len(data_module.roman_vocab) # English vocab size
    output_vocab_size = len(data_module.native_vocab) # Indian language vocab size

    model = Seq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        rnn_cell=config.rnn_cell,
        num_layers=config.num_layers,
        learning_rate=config.learning_rate,
        dropout=config.dropout,
        is_attentive=config.attentive,
        train_teacher_forcing_ratio=config.train_teacher_forcing_ratio,
    )

    # Initialize the PyTorch Lightning Trainer
    logger = WandbLogger(project=config.project)
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=logger, 
        accelerator='auto',
        devices='auto',
        gradient_clip_val=1.0,  # Prevent exploding gradients
        enable_checkpointing=False,
    )

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    sweep_configuration = {
        'method': 'bayes',
        'name': 'DA24S023_DA6401_A3',
        'metric': {'name': 'val/word_accuracy', 'goal': 'maximize'},
        'parameters': {
            'embedding_dim': {'values': [192, 256, 384, 512]},
            'hidden_dim': {'values': [384, 512, 768, 1024]},
            'num_layers': {'values': [4, 6, 8, 10]},
            'rnn_cell': {'values': ['RNN', 'LSTM', 'GRU']},
            'dropout': {'values': [0.2, 0.3, 0.4]},
            'learning_rate': {'values': [5e-3, 1e-3, 5e-4]},
            'train_teacher_forcing_ratio': {'values': [0.3, 0.5, 0.7]},
        }
    }

    parser = argparse.ArgumentParser(description="Train Seq2Seq model with Wandb sweeps")
    parser.add_argument(
        '-sc', '--sweep_count', 
        type=int, 
        default=50, 
        help='Number of Wandb sweeps (default: 30)'
    )
    parser.add_argument(
        '-l', '--lang',
        type=str,
        default='ta',
        choices=['hi', 'bn', 'gu', 'kn', 'ml', 'or', 'pa', 'ta', 'te'],  # Languages available in Dakshina dataset
        help='Language to train (default: ta). Choices: hi, bn, gu, kn, ml, or, pa, ta, te'
    )
    parser.add_argument(
        '-p', '--project',
        type=str,
        default='DA24S023_DA6401_A3',
        help='Wandb project name (default: DA24S023_DA6401_A3)'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=512,
        help='Batch size for training (default: 512)'
    )
    parser.add_argument(
        '-nw', '--num_workers',
        type=int,
        default=32,
        help='Number of workers for data loading (default: 32)'
    )
    parser.add_argument(
        '-ep', '--epochs',
        type=int,
        default=20,
        help='Number of epochs for training (default: 20)'
    )
    parser.add_argument(
        '-atn', '--attentive',
        type=bool,
        default=False,
        help='Use Luong attention mechanism in the decoder (default: False)'
    )
    args = parser.parse_args()

    sweep_configuration['parameters']['lang'] = {'values': [args.lang]}
    sweep_configuration['parameters']['project'] = {'values': [args.project]}
    sweep_configuration['parameters']['batch_size'] = {'values': [args.batch_size]}
    sweep_configuration['parameters']['num_workers'] = {'values': [args.num_workers]}
    sweep_configuration['parameters']['epochs'] = {'values': [args.epochs]}
    sweep_configuration['parameters']['attentive'] = {'values': [args.attentive]}
    sweep_configuration['name'] = f"{args.project}_attn:{args.attentive}"

    sweep_id = wandb.sweep(sweep_configuration, project=args.project)
    wandb.agent(sweep_id, function=hparam_search, count=args.sweep_count)
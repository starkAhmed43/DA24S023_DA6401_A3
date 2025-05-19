import yaml
import wandb
import argparse
from pathlib import Path
from model import Seq2SeqModel
import pytorch_lightning as pl
from datamodule import DakshinaDataModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42, workers=True)

def parse_args():
    """Parse command line arguments and YAML config file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--lang', type=str, default='ta')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden_dim', type=int, default=384)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--project', type=str, default='DA24S023_DA6401_A3')
    parser.add_argument('--rnn_cell', type=str, default='GRU')
    parser.add_argument('--attention_type', type=str, default='none', choices=['luong', 'bahdanau', 'none'])
    parser.add_argument('--train_teacher_forcing_ratio', type=float, default=0.5)
    return parser.parse_args()

def load_config(args):
    """Load configuration from YAML file if provided."""
    if args.config:
        if not Path(args.config).is_file():
            raise FileNotFoundError(f"Config file {args.config} not found.")
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # YAML overrides argparse
        for k, v in yaml_config.items():
            setattr(args, k, v)
    # Convert 'none' to None for attention_type
    if getattr(args, 'attention_type', 'none') == 'none':
        args.attention_type = None
    return args

def make_run_name(args):
    # Compose a unique run name from all key params
    run_name = (
        f"lang:{args.lang}_bs:{args.batch_size}_edim:{args.embedding_dim}_hdim:{args.hidden_dim}_"
        f"nl:{args.num_layers}_dr:{args.dropout}_lr:{args.learning_rate}_"
        f"rnn:{args.rnn_cell}_attn:{args.attention_type or 'none'}_tfr:{args.train_teacher_forcing_ratio}"
    )
    return run_name

def train(args):
    """
    Main function to train the model.
    This function initializes the Wandb logger, sets up the data module,
    initializes the model, and starts the training process.
    """
    run_name = make_run_name(args)

    # Initialize Wandb
    wandb.init(
        project=args.project,
        name=run_name,
        config=vars(args),
    )

    # Instantiate DataModule for the chosen language (For example, 'ta' for Tamil)
    data_module = DakshinaDataModule(lang=args.lang, batch_size=args.batch_size, num_workers=args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    # Instantiate the model
    input_vocab_size = len(data_module.roman_vocab) # English vocab size
    output_vocab_size = len(data_module.native_vocab) # Indian language vocab size

    model = Seq2SeqModel(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        rnn_cell=args.rnn_cell,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        attention_type=args.attention_type,
        train_teacher_forcing_ratio=args.train_teacher_forcing_ratio,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=run_name,
        save_top_k=1,
        monitor="val/word_accuracy",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Initialize the PyTorch Lightning Trainer
    logger = WandbLogger(project=args.project, name=run_name)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger, 
        accelerator='auto',
        devices='auto',
        gradient_clip_val=1.0,  # Prevent exploding gradients
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save checkpoint locally and to wandb
    best_ckpt_path = checkpoint_callback.best_model_path
    wandb.save(best_ckpt_path)
    print(f"Best checkpoint saved at: {best_ckpt_path}")

if __name__ == "__main__":
    args = parse_args()
    args = load_config(args)

    train(args)
    
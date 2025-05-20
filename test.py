import io
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pathlib import Path
from model import Seq2SeqModel
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datamodule import DakshinaDataModule
from pytorch_lightning.loggers.wandb import WandbLogger
from train import parse_args, load_config, make_run_name

tamil_font = fm.FontProperties(fname="./fonts/NotoSansTamil-Regular.ttf")

def plot_attn_heatmap(attn_weights, all_inputs, all_preds, all_refs, idx2char_native, idx2char_roman):
    sns.set_theme()
    n_grid = 9

    # Decode all predictions and references for comparison
    decoded_preds = [
        ''.join([idx2char_native.get(i.item(), '') for i in pred if i.item() not in (0, 1, 2)])
        for pred in all_preds
    ]
    decoded_refs = [
        ''.join([idx2char_native.get(i.item(), '') for i in ref if i.item() not in (0, 1, 2)])
        for ref in all_refs
    ]

    # Find indices of correct and incorrect predictions
    correct_indices = [i for i, (p, r) in enumerate(zip(decoded_preds, decoded_refs)) if p == r]
    incorrect_indices = [i for i, (p, r) in enumerate(zip(decoded_preds, decoded_refs)) if p != r]

    # Select up to 6 correct and 3 incorrect
    selected_correct = np.random.choice(correct_indices, min(6, len(correct_indices)), replace=False) if correct_indices else []
    selected_incorrect = np.random.choice(incorrect_indices, min(3, len(incorrect_indices)), replace=False) if incorrect_indices else []
    indices = list(selected_correct) + list(selected_incorrect)
    # If not enough, fill with random
    if len(indices) < n_grid:
        remaining = list(set(range(len(attn_weights))) - set(indices))
        indices += list(np.random.choice(remaining, n_grid - len(indices), replace=False))

    heatmap_imgs = []
    for idx in indices:
        attn = attn_weights[idx].numpy()
        inp = all_inputs[idx]
        pred = all_preds[idx]
        ref = all_refs[idx]
        inp_str = ''.join([idx2char_roman.get(i.item(), '') for i in inp if i.item() not in (0, 1, 2)])
        pred_str = decoded_preds[idx]
        ref_str = decoded_refs[idx]
        correct = pred_str == ref_str

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            attn[:len(pred_str), :len(inp_str)],
            ax=ax,
            cmap='rocket',
            cbar=True,
            cbar_kws={"shrink": 0.7, "label": "Attention"}
        )
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.set_xticks(np.arange(len(inp_str)) + 0.5)
        ax.set_xticklabels(list(inp_str), fontproperties=None, rotation=0)
        ax.set_yticks(np.arange(len(pred_str)) + 0.5)
        ax.set_yticklabels(list(pred_str), fontproperties=tamil_font, rotation=0)
        ax.set_title(
            f"{'Correct' if correct else 'Incorrect'}\nInput: {inp_str}\nPred: {pred_str}\nGT: {ref_str}",
            fontproperties=tamil_font, fontsize=10
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        heatmap_imgs.append(wandb.Image(img, caption=f"{'Correct' if correct else 'Incorrect'}\nInput: {inp_str}\nPred: {pred_str}\nGT: {ref_str}",))
        plt.close(fig)

    wandb.log({"Attention Heatmaps Panel": heatmap_imgs})

def test(args):
    """
    Main function to test the model.
    """

    if args.attention_type == None:
        output_path = Path("./predictions_vanilla")
    elif args.attention_type == 'bahdanau':
        output_path = Path("./predictions_attention_bahdanau")
    elif args.attention_type == 'luong':
        output_path = Path("./predictions_attention_luong")
    else:
        raise ValueError(f"Invalid attention type: {args.attention_type}")
    output_path.mkdir(parents=True, exist_ok=True)

    ckpt_file = Path("./data/checkpoints") / f"{make_run_name(args)}.ckpt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file {ckpt_file} not found.")
    
    # Load data
    data_module = DakshinaDataModule(lang=args.lang, batch_size=args.batch_size, num_workers=args.num_workers)
    data_module.prepare_data()
    data_module.setup(stage='test')

    # Get vocabularies for decoding
    idx2char_native = {v: k for k, v in data_module.native_vocab.items()}
    idx2char_roman = {v: k for k, v in data_module.roman_vocab.items()}

    # Load model
    model = Seq2SeqModel.load_from_checkpoint(ckpt_file)

    # Initialize Wandb
    wandb.init(project=args.project, name=f"test_{make_run_name(args)}", config=vars(args))
    logger = WandbLogger(
        project=args.project,
        name=f"test_{make_run_name(args)}",
        config=vars(args),
    )
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='auto', 
        devices='auto',
        logger=logger)

    # Run test
    trainer.test(model, datamodule=data_module)
    results = model.test_results

    all_inputs, all_preds, all_refs = results["input_seq"], results["preds"], results["target_seq"]    
    decoded_inputs, decoded_preds, decoded_refs = [], [], []

    for inp, pred, ref in zip(all_inputs, all_preds, all_refs):
        inp_str = ''.join([idx2char_roman.get(i.item(), '') for i in inp if i.item() not in (0, 1, 2)])
        pred_str = ''.join([idx2char_native.get(i.item(), '') for i in pred if i.item() not in (0, 1, 2)])
        ref_str = ''.join([idx2char_native.get(i.item(), '') for i in ref if i.item() not in (0, 1, 2)])
        decoded_inputs.append(inp_str)
        decoded_preds.append(pred_str)
        decoded_refs.append(ref_str)
    correct_col = ["Correct" if p == r else "Incorrect" for p, r in zip(decoded_preds, decoded_refs)]

    df = pd.DataFrame({
        'Romanized Input': decoded_inputs,
        'Ground Truth': decoded_refs,
        'Predicted Output': decoded_preds,
        'Correct': correct_col,
    })
    df.to_csv(output_path / "test_results.csv", index=False, sep=",")
    df.to_csv(output_path / "test_results.tsv", index=False, sep="\t")

    sample_df = df.sample(20, random_state=42)
    table = wandb.Table(columns=["Romanized Input", "Ground Truth", "Predicted Output", "Correct"])
    for _, row in sample_df.iterrows():
        table.add_data(row['Romanized Input'], row['Ground Truth'], row['Predicted Output'], row['Correct'])
    wandb.log({"Test Predictions Grid": table})

    # Log attention heatmaps
    if args.attention_type in ["bahdanau", "luong"]:
        attn_weights = results["attn_weights"]
        plot_attn_heatmap(attn_weights, all_inputs, all_preds, all_refs, idx2char_native, idx2char_roman)

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    args = load_config(args)
    
    test(args)
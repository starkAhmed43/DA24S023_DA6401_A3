import pandas as pd
from pathlib import Path
from model import Seq2SeqModel
import pytorch_lightning as pl
from datamodule import DakshinaDataModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from train import parse_args, load_config, make_run_name

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

    ckpt_file = Path("./checkpoints") / f"{make_run_name(args)}.ckpt"
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
    
    # Create trainer
    trainer = pl.Trainer(accelerator='auto', devices='auto')

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
    
    df = pd.DataFrame({
        'Romanized Input': decoded_inputs,
        'Ground Truth': decoded_refs,
        'Predicted Output': decoded_preds,
    })
    df.to_csv(output_path / "test_results.csv", index=False, sep=",")
    df.to_csv(output_path / "test_results.tsv", index=False, sep="\t")
    df.to_excel(output_path / "test_results.xlsx", index=False)

if __name__ == "__main__":
    args = parse_args()
    args = load_config(args)
    
    test(args)
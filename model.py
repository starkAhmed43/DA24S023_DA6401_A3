import wandb
import torch
import pandas as pd
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl

class Seq2SeqModel(pl.LightningModule):
    def __init__(self, 
        input_vocab_size, 
        output_vocab_size, 
        embedding_dim=128, 
        hidden_dim=256, 
        rnn_cell='LSTM', 
        num_layers=4, 
        learning_rate=1e-3, 
        dropout=0.1,
        attention_type=None,
        train_teacher_forcing_ratio=0.5,
    ):
        """
        A PyTorch Lightning-based sequence-to-sequence model for transliteration.
        Args:
            input_vocab_size (int): Size of the input vocabulary (Latin characters).
            output_vocab_size (int): Size of the output vocabulary (Devanagari characters).
            embedding_dim (int): Dimension of the character embeddings. Default: 128.
            hidden_dim (int): Dimension of the hidden states in the RNNs. Default: 256.
            rnn_cell (str): Type of RNN cell ('RNN', 'LSTM', 'GRU'). Default: 'LSTM'.
            num_layers (int): Number of layers in both the encoder and decoder. Default: 4.
            learning_rate (float): Learning rate for the optimizer. Default: 1e-3.
            dropout (float): Dropout rate for the RNN layers. Default: 0.1.
            attention_type (str): Type of attention mechanism ('bahdanau', 'luong', or None). Default: None.
            teacher_forcing_ratio (float): Probability of using teacher forcing during training. Default: 0.5.
        """
        super(Seq2SeqModel, self).__init__()
        self.save_hyperparameters()

        self.attention_type = attention_type
        self.train_teacher_forcing_ratio = train_teacher_forcing_ratio

        # Embedding layers for input and output vocabularies
        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_dim)

        # Select the RNN cell type
        rnn_cell_map = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        if rnn_cell not in rnn_cell_map:
            raise ValueError("Invalid RNN cell type. Choose from 'RNN', 'LSTM', 'GRU'.")
        RNNCell = rnn_cell_map[rnn_cell]

        # Bahdanau attention layers (only if needed)
        if attention_type == 'bahdanau':
            self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim)
            self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

        # Encoder RNN
        self.encoder = RNNCell(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Decoder RNN
        decoder_input_size = embedding_dim + (hidden_dim if attention_type else 0)
        self.decoder = RNNCell(
            input_size=decoder_input_size, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully connected layer to map decoder outputs to output vocabulary
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

        # Store RNN cell type for forward pass
        self.rnn_cell = rnn_cell

        # Metrics
        self.train_accuracy = tm.Accuracy(task='multiclass', num_classes=output_vocab_size)
        self.val_accuracy = tm.Accuracy(task='multiclass', num_classes=output_vocab_size)
        self.test_accuracy = tm.Accuracy(task='multiclass', num_classes=output_vocab_size)

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5, return_attn=False):
        """
        Forward pass for the seq2seq model.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, input_seq_len).
            target_seq (Tensor, optional): Target sequence of shape (batch_size, target_seq_len).
            teacher_forcing_ratio (float, optional): Probability of using teacher forcing during training.
            return_attn (bool, optional): Whether to return attention weights. Default: False.

        Returns:
            Tensor: Output logits of shape (batch_size, target_seq_len, output_vocab_size).
        """
        batch_size = input_seq.size(0)
        target_seq_len = target_seq.size(1) if target_seq is not None else 20  # Default max length for inference

        # Encode the input sequence
        input_embedded = self.input_embedding(input_seq)  # (batch_size, input_seq_len, embedding_dim)
        encoder_outputs, hidden = self.encoder(input_embedded)  # hidden: (num_layers, batch_size, hidden_dim) or tuple for LSTM

        # Prepare the initial input for the decoder (start token, assumed to be index 0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=input_seq.device)  # (batch_size, 1)
        decoder_hidden = hidden

        # Decode the sequence
        outputs = []
        attn_weights_all = [] if self.attention_type else None
        for t in range(target_seq_len):
            decoder_embedded = self.output_embedding(decoder_input)  # (batch_size, 1, embedding_dim)
            if self.attention_type == 'luong':
                dec_h = decoder_hidden[0][-1] if self.rnn_cell == 'LSTM' else decoder_hidden[-1]

                # Compute attention scores
                attn_scores = torch.bmm(
                    encoder_outputs,  # (batch, src_len, hidden_dim)
                    dec_h.unsqueeze(2)  # (batch, hidden_dim, 1)
                ).squeeze(2)  # (batch, src_len)                
                attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, src_len)
                attn_weights_all.append(attn_weights.unsqueeze(1))  # Store attention weights for visualization

                # Compute context vector
                context = torch.bmm(
                    attn_weights.unsqueeze(1),  # (batch, 1, src_len)
                    encoder_outputs  # (batch, src_len, hidden_dim)
                )  # (batch, 1, hidden_dim)
                decoder_embedded = torch.cat((decoder_embedded, context), dim=2)  # (batch_size, 1, embedding_dim + hidden_dim)
            elif self.attention_type == 'bahdanau':
                dec_h = decoder_hidden[0][-1] if self.rnn_cell == 'LSTM' else decoder_hidden[-1]
                dec_h_exp = dec_h.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # (batch, src_len, hidden_dim)
                
                # Concatenate encoder outputs and expanded decoder hidden state
                concat = torch.cat((encoder_outputs, dec_h_exp), dim=2)  # (batch, src_len, hidden_dim + embedding_dim)

                # Apply the attention linear layer
                energy = torch.tanh(self.attn_W(concat))  # (batch, src_len, hidden_dim)

                # Compute the attention scores
                attn_scores = self.attn_v(energy).squeeze(2)  # (batch, src_len)

                # Compute attention weights
                attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, src_len)
                attn_weights_all.append(attn_weights.unsqueeze(1))  # Store attention weights for visualization

                # Compute context vector
                context = torch.bmm(
                    attn_weights.unsqueeze(1),  # (batch, 1, src_len)
                    encoder_outputs  # (batch, src_len, hidden_dim)
                )  # (batch, 1, hidden_dim)
                decoder_embedded = torch.cat((decoder_embedded, context), dim=2)  # (batch_size, 1, embedding_dim + hidden_dim)
            else:
                pass

            # Pass through the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_embedded, decoder_hidden)

            # Map decoder output to vocabulary space
            logits = self.fc(decoder_output.squeeze(1))  # (batch_size, output_vocab_size)
            outputs.append(logits.unsqueeze(1))  # Collect outputs

            # Determine the next input for the decoder
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t].unsqueeze(1)  # Teacher forcing
            else:
                decoder_input = logits.argmax(dim=1, keepdim=True)  # Predicted token

        # Concatenate outputs along the time dimension
        outputs = torch.cat(outputs, dim=1)  # (batch_size, target_seq_len, output_vocab_size)
        preds = outputs.argmax(dim=2)  # (batch_size, target_seq_len)
        if return_attn and self.attention_type:
            attn_tensor = torch.cat(attn_weights_all, dim=1)  # (batch, tgt_len, src_len)
            return outputs, preds, attn_tensor
        return outputs, preds

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (tuple): A batch of data containing input and target sequences.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Character-wise training loss.
        """
        native_word, romanization, attestation = batch
        input_seq, target_seq = romanization, native_word

        output_logits, preds = self(input_seq, target_seq, self.train_teacher_forcing_ratio)
        loss = nn.CrossEntropyLoss()(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))

        # Use attestation as weights in loss calculation
        weights = attestation.float().repeat_interleave(target_seq.size(1))
        weighted_loss = (loss * weights).mean()

        word_matches = (preds == target_seq).all(dim=1).float()  # shape: (batch_size,)
        
        self.train_accuracy(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        self.log("train/char_accuracy", self.train_accuracy, on_epoch=True, on_step=False)
        self.log("train/word_accuracy", word_matches.mean(), on_epoch=True, on_step=False)
        self.log("train/loss", weighted_loss, on_epoch=True, on_step=False)
        return weighted_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model. 

        Args:
            batch (tuple): A batch of data containing input and target sequences.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Validation loss.
        """
        native_word, romanization, attestation = batch
        input_seq, target_seq = romanization, native_word

        # Use teacher forcing ratio of 0 for validation
        output_logits, preds = self(input_seq, target_seq, teacher_forcing_ratio=0.0)
        loss = nn.CrossEntropyLoss()(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        
        weights = attestation.float().repeat_interleave(target_seq.size(1))
        weighted_loss = (loss * weights).mean()
        
        word_matches = (preds == target_seq).all(dim=1).float()

        self.val_accuracy(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        self.log("val/char_accuracy", self.val_accuracy, on_epoch=True, on_step=False)
        self.log("val/word_accuracy", word_matches.mean(), on_epoch=True, on_step=False)
        self.log("val/loss", weighted_loss, on_epoch=True, on_step=False)
        return weighted_loss

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.

        Args:
            batch (tuple): A batch of data containing input and target sequences.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Test loss.
        """
        native_word, romanization, attestation = batch
        input_seq, target_seq = romanization, native_word

        # Use teacher forcing ratio of 0 for testing
        if self.attention_type in ['bahdanau', 'luong']:
            output_logits, preds, attn_weights = self(input_seq, target_seq, teacher_forcing_ratio=0.0, return_attn=True)
        else:
            output_logits, preds = self(input_seq, target_seq, teacher_forcing_ratio=0.0)
            attn_weights = None
        
        loss = nn.CrossEntropyLoss()(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        weights = attestation.float().repeat_interleave(target_seq.size(1))
        weighted_loss = (loss * weights).mean()        
        word_matches = (preds == target_seq).all(dim=1).float()        
        self.test_accuracy(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        
        return {
            "loss": weighted_loss.detach().cpu(),
            "word_accuracy": word_matches.mean().detach().cpu(),
            "char_accuracy": self.test_accuracy.compute().detach().cpu(),
            "input_seq": input_seq.detach().cpu(),
            "preds": preds.detach().cpu(),
            "target_seq": target_seq.detach().cpu(),
            "attn_weights": attn_weights.detach().cpu() if attn_weights is not None else None,
        }
    
    def on_test_epoch_start(self):
        self._test_outputs = []

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        self._test_outputs.append(outputs)

    def on_test_epoch_end(self):
        all_inputs, all_preds, all_refs, all_attns = [], [], [], []
        all_char_acc, all_word_acc, all_loss = [], [], []
        for o in self._test_outputs:
            all_inputs.extend([seq for seq in o["input_seq"]])
            all_preds.extend([seq for seq in o["preds"]])
            all_refs.extend([seq for seq in o["target_seq"]])
            all_attns.extend([a for a in o["attn_weights"]]) if o["attn_weights"] is not None else None
            
            all_char_acc.append(o["char_accuracy"])
            all_word_acc.append(o["word_accuracy"])
            all_loss.append(o["loss"])
        
        self.test_results = {
            "input_seq": all_inputs,
            "preds": all_preds,
            "target_seq": all_refs,
            "attn_weights": all_attns,
        }

        metrics = {
            "char_accuracy": torch.stack(all_char_acc).mean().item(),
            "word_accuracy": torch.stack(all_word_acc).mean().item(),
            "loss": torch.stack(all_loss).mean().item(),
        }
        df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        table = wandb.Table(dataframe=df)
        wandb.log({"test/metrics_table": table})
        

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            Optimizer: Optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
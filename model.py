import torch
import torch.nn as nn
import torchmetrics as tm
import pytorch_lightning as pl

class Seq2SeqModel(pl.LightningModule):
    """
    A PyTorch Lightning-based sequence-to-sequence model for transliteration.

    Args:
        input_vocab_size (int): Size of the input vocabulary (Latin characters).
        output_vocab_size (int): Size of the output vocabulary (Devanagari characters).
        embedding_dim (int): Dimension of the character embeddings.
        hidden_dim (int): Dimension of the hidden states in the RNNs.
        rnn_cell (str): Type of RNN cell ('RNN', 'LSTM', 'GRU').
        num_layers (int): Number of layers in both the encoder and decoder.
        learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim=128, hidden_dim=256, rnn_cell='LSTM', num_layers=1, learning_rate=1e-3, dropout=0.1):
        """
        Initializes the Seq2SeqModel.
        """
        super(Seq2SeqModel, self).__init__()
        self.save_hyperparameters()

        # Embedding layers for input and output vocabularies
        self.input_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(output_vocab_size, embedding_dim)

        # Select the RNN cell type
        rnn_cell_map = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}
        if rnn_cell not in rnn_cell_map:
            raise ValueError("Invalid RNN cell type. Choose from 'RNN', 'LSTM', 'GRU'.")
        RNNCell = rnn_cell_map[rnn_cell]

        # Encoder RNN
        self.encoder = RNNCell(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # Decoder RNN
        self.decoder = RNNCell(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # Fully connected layer to map decoder outputs to output vocabulary
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

        # Store RNN cell type for forward pass
        self.rnn_cell = rnn_cell

        # Metrics
        self.train_accuracy = tm.Accuracy(task='multiclass', num_classes=output_vocab_size)
        self.val_accuracy = tm.Accuracy(task='multiclass', num_classes=output_vocab_size)
        self.test_accuracy = tm.Accuracy(task='multiclass', num_classes=output_vocab_size)

    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5):
        """
        Forward pass for the seq2seq model.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, input_seq_len).
            target_seq (Tensor, optional): Target sequence of shape (batch_size, target_seq_len).
            teacher_forcing_ratio (float, optional): Probability of using teacher forcing during training.

        Returns:
            Tensor: Output logits of shape (batch_size, target_seq_len, output_vocab_size).
        """
        batch_size = input_seq.size(0)
        target_seq_len = target_seq.size(1) if target_seq is not None else 20  # Default max length for inference

        # Encode the input sequence
        input_embedded = self.input_embedding(input_seq)  # (batch_size, input_seq_len, embedding_dim)
        _, hidden = self.encoder(input_embedded)  # hidden: (num_layers, batch_size, hidden_dim) or tuple for LSTM

        # Prepare the initial input for the decoder (start token, assumed to be index 0)
        decoder_input = torch.zeros(batch_size, 1, dtype=torch.long, device=input_seq.device)  # (batch_size, 1)
        decoder_hidden = hidden

        # Decode the sequence
        outputs = []
        for t in range(target_seq_len):
            decoder_embedded = self.output_embedding(decoder_input)  # (batch_size, 1, embedding_dim)
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

        output_logits, preds = self(input_seq, target_seq, teacher_forcing_ratio=0.5)
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
        input_seq, target_seq = batch
        output_logits = self(input_seq, target_seq, teacher_forcing_ratio=0.0)
        loss = nn.CrossEntropyLoss()(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        
        self.test_accuracy(output_logits.view(-1, self.hparams.output_vocab_size), target_seq.view(-1))
        self.log("test/accuracy", self.test_accuracy)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            Optimizer: Optimizer for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
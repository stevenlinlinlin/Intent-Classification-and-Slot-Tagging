from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        device: str,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # model architecture
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_class = num_class
        self.embed_dim = 300

        self.rnn = torch.nn.LSTM(300,hidden_size,num_layers=num_layers,bidirectional=bidirectional, dropout=dropout, batch_first=True)

        self.fc1 = torch.nn.Linear(2*hidden_size,num_class) # 2 => bidirectional=True

        self.sigmoid = torch.nn.Sigmoid()


    @property
    def encoder_output_size(self) -> int:
        # calculate the output dimension of rnn
        return self.hidden_size * 2

    def forward(self, batch, text_lengths) -> torch.Tensor:
        # implement model forward
        embed_batch = self.embed(batch)

        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embed_batch, text_lengths, batch_first=True)
        _, (hidden,_) = self.rnn(packed_embedded)

        output = self.fc1(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        output = self.sigmoid(output)

        return output


class SeqTagger(SeqClassifier):
    def forward(self, batch, text_lengths) -> torch.Tensor:
        # implement model forward
        embed_batch = self.embed(batch)
        
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embed_batch, text_lengths, batch_first=True)
        out, (_,_) = self.rnn(packed_embedded)
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        output = self.fc1(seq_unpacked)

        output = self.sigmoid(output)

        return output

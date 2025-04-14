import torch
import torch.nn as nn

CONFIGS = {
    "lstm_small": {"embed_dim": 64, "hidden_dim": 128, "num_layers": 1},
    "lstm_medium": {"embed_dim": 256, "hidden_dim": 512, "num_layers": 2},
    "lstm_large": {"embed_dim": 512, "hidden_dim": 1024, "num_layers": 3},
}

def get_model(config, vocab_size, num_classes, seq2seq=False): 
    return cLSTM(vocab_size, config["embed_dim"], config["hidden_dim"], num_classes, config["num_layers"], seq2seq)

class cLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, seq2seq=False):
        super(cLSTM, self).__init__()

        self.seq2seq = seq2seq
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x["input_ids"].squeeze())
        lstm_out, _ = self.lstm(x)
        
        if self.seq2seq:
            output = self.fc(lstm_out[:, -1, :])  # Shape: (batch, seq_len)
        else:
            output = self.fc(lstm_out[:, -1, :])  # Shape: (batch, num_classes)
            output = self.sigmoid(output)

        return output

class LSTMWrapper(nn.Module): 
    def __init__(self, size, num_classes, seq2seq=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if size not in CONFIGS:
            raise AttributeError(f"Missing size {size}")
        self.vocab_size = 30522  # Replace with tokenizer vocab if needed
        self.model = get_model(CONFIGS[size], self.vocab_size, num_classes, seq2seq=seq2seq)

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {pytorch_total_params}")

    def forward(self, x, y=None): 
        x = self.model(x)# * self.vocab_size
        return x, None
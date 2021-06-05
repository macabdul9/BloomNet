import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, dropout=0.10, num_layers=1, bidirectional=True, num_classes=6):
        """LSTM baseline for bloom_taxonomy classification

        Args:
            vocab_size (int): size of vocabulary
            hidden_size (int): hidden size, same as embedding size
            dropout (float, optional): dropout rat. Defaults to 0.10.
            n_layers (int, optional): how many layers in LSTM. Defaults to 1.
            bidirectional (bool, optional): whether bidirectional or not. Defaults to True.
            num_classes (int, optional): no of classes in training set. Defaults to 6.
        """
        
        super(LSTMModel, self).__init__()
        
        
        # embedding layer to get embeddings of tokenized input
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # classification module
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=hidden_size, out_features=hidden_size//2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size//2, out_features=num_classes)
        ])
        
    def forward(self, input_ids, _len=None, attention_mask=None):
        
        """forward function of LSTM model.

        Args:
            input_ids (Tensor): shape = [batch_size, max_len]
            _len (Tensor, optional): shape = [batch]. Defaults to None.
            attention_mask ([type], optional): shape = [batch_size, max_len]. Defaults to None.
        """
        
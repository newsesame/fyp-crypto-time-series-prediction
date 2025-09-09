import torch
import torch.nn as nn
from torch import Tensor


class SineActivation(nn.Module):
    def __init__(self, in_features, periodic_features, out_features, dropout):
        super(SineActivation, self).__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, out_features - in_features - periodic_features))
        self.b0 = nn.parameter.Parameter(torch.randn(1, out_features - in_features - periodic_features))
        self.w = nn.parameter.Parameter(torch.randn(in_features, periodic_features))
        self.b = nn.parameter.Parameter(torch.randn(1, periodic_features))
        self.activation = torch.sin
        self.dropout = nn.Dropout(dropout)

    def Time2Vector(self, data):
        v_linear = torch.matmul(self.w0.t(), data.transpose(1, 2)).transpose(1, 2) + self.b0
        v_sin = self.activation(torch.matmul(self.w.t(), data.transpose(1, 2)).transpose(1, 2) + self.b)
        data = torch.cat([v_linear, v_sin, data], 2)
        return data

    def forward(self, data):
        data = self.Time2Vector(data)
        data = self.dropout(data)
        return data


# ==================== Regression Models ====================

class BTC_Transformer(nn.Module):
    """Transformer regression model for Bitcoin price prediction
    
    This model uses a Transformer architecture with sine activation for time series
    feature encoding and a sequence-to-sequence structure for price prediction.
    """
    
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 in_features: int,
                 periodic_features: int,
                 out_features: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(BTC_Transformer, self).__init__()

        self.sine_activation = SineActivation(in_features=in_features,
                                              periodic_features=periodic_features,
                                              out_features=out_features,
                                              dropout=dropout)

        self.transformer = nn.Transformer(d_model=out_features,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          activation=activation)

        self.generator = nn.Linear(out_features, in_features)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.sine_activation(src), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.sine_activation(tgt), memory, tgt_mask)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor = None,
                tgt_mask: Tensor = None,
                mem_mask: Tensor = None,
                src_padding_mask: Tensor = None,
                tgt_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None):

        src_emb = self.sine_activation(src)
        tgt_emb = self.sine_activation(trg)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, mem_mask,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)


class GRUSeq2Seq(nn.Module):
    """GRU sequence-to-sequence regression model for Bitcoin price prediction
    
    This model uses a GRU-based encoder-decoder architecture for time series
    price prediction with configurable hidden size and number of layers.
    """
    
    def __init__(self, in_features: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.decoder = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.proj = nn.Linear(hidden_size, in_features)

    def forward(self, src: Tensor, trg: Tensor, *_, **__):
        # src: (sequence_length, batch_size, features), trg: (target_length, batch_size, features)
        _, h = self.encoder(src)
        out, _ = self.decoder(trg, h)
        y = self.proj(out)
        return y


# ==================== Classification Models ====================

class TransformerClassifier(nn.Module):
    """Transformer classifier for predicting price movement direction (+1 or -1)
    
    This model uses a Transformer encoder with sine activation for time series
    feature encoding and a classification head for binary price direction prediction.
    """

    def __init__(self,
                 num_encoder_layers: int,
                 in_features: int,
                 periodic_features: int,
                 out_features: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super(TransformerClassifier, self).__init__()

        self.sine_activation = SineActivation(in_features=in_features,
                                              periodic_features=periodic_features,
                                              out_features=out_features,
                                              dropout=dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=out_features,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False
            ),
            num_layers=num_encoder_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_features, out_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features // 2, 2)  # 2 classes: +1 (up), -1 (down)
        )

    def forward(self, x: Tensor):
        # x: (batch, seq_len, features) -> transpose to (seq_len, batch, features)
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        x_emb = self.sine_activation(x)
        encoded = self.transformer(x_emb)
        # Use the output from the last time step
        last_output = encoded[-1]  # (batch, out_features)
        logits = self.classifier(last_output)
        return logits


class GRUClassifier(nn.Module):
    """GRU classifier for predicting price movement direction (+1 or -1)
    
    This model uses a GRU-based architecture for time series feature encoding
    and a classification head for binary price direction prediction.
    """

    def __init__(self, in_features: int, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # 2 classes: +1 (up), -1 (down)
        )

    def forward(self, x: Tensor):
        # x: (batch, seq_len, features) -> transpose to (seq_len, batch, features)
        x = x.transpose(0, 1)  # (seq_len, batch, features)
        _, h = self.gru(x)
        # Use the last hidden state from the last layer
        last_hidden = h[-1]  # (batch, hidden_size)
        logits = self.classifier(last_hidden)
        return logits


# ==================== Exports ====================

__all__ = [
    "SineActivation",
    # Regression models
    "BTC_Transformer", 
    "GRUSeq2Seq",
    # Classification models
    "TransformerClassifier", 
    "GRUClassifier"
]

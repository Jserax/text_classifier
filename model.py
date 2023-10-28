import torch
from torch import nn
from transformers import DistilBertModel


class Model(nn.Module):
    def __init__(
        self,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
            "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(264, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.bert(x, mask)[0][:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        out = self.activation(x)
        return out.squeeze()

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def prepare_data() -> pd.DataFrame:
    with open("data/dataset.txt", "r", encoding="utf-8") as file:
        comments = file.readlines()
    comment = [" ".join(txt.split(" ")[1:]) for txt in comments]
    toxic = [0 if txt.split(" ")[0] == "__label__NORMAL" else 1 for txt in comments]
    ds1 = pd.DataFrame({"comment": comment, "toxic": toxic})
    ds2 = pd.read_csv("data/labeled.csv")
    ds = pd.concat([ds1, ds2])
    return ds


def prepare_loader(
    data: pd.DataFrame,
    max_len: int = 128,
    batch_size: int = 128,
    test_size: float = 0.1,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train, val = train_test_split(
        data, test_size=test_size, stratify=data["toxic"], random_state=42
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepPavlov/distilrubert-tiny-cased-conversational-v1"
    )
    x = tokenizer.batch_encode_plus(
        train.comment.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    y = torch.tensor(data=train.toxic.values, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        list(zip(x.input_ids, x.attention_mask, y)),
        shuffle=True,
        batch_size=batch_size,
    )

    x = tokenizer.batch_encode_plus(
        val.comment.tolist(),
        add_special_tokens=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    y = torch.tensor(data=val.toxic.values, dtype=torch.float32)
    val_loader = torch.utils.data.DataLoader(
        list(zip(x.input_ids, x.attention_mask, y)),
        shuffle=False,
        batch_size=batch_size,
    )

    return train_loader, val_loader

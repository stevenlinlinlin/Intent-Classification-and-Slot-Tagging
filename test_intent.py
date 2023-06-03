import json
import csv
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)

    # crecate DataLoader for test dataset
    ### need to solve last batch problem
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn) 

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.device
    ).to(args.device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters())

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    epoch = ckpt['epoch']
    loss = ckpt['loss']

    # predict dataset
    pred_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            inputs = torch.tensor(batch["text"], dtype=torch.int32).to(args.device)

            predicts = model(inputs, batch["text_lengths"])
            
            tmp_list = [[0]] * args.batch_size
            for i in range(predicts.size(0)):
                tmp_list[batch["unsort_id"][i]] = [batch["id"][i], dataset.idx2label(torch.argmax(predicts[i]).item())]
            pred_list.append(tmp_list)

    # write prediction to file (args.pred_file)
    with open(args.pred_file, "w", newline='') as f:
        ans = csv.writer(f)
        ans.writerow(["id", "intent"])
        for batch in pred_list:
            for data in batch:
                if len(data) > 1:
                    ans.writerow(data)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

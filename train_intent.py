import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from this import d
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from model import SeqClassifier
from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)  # length:6491
    
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn, drop_last=True, shuffle=True)
    dev_dataloader = DataLoader(datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn, drop_last=True, shuffle=True)
    #for data in train_dataloader:
        #print(type(data))
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")     # torch.Size([6491, 300])

    # init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(intent2idx), args.device).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss().to(args.device) # BCEloss -> CrossEntropyloss

    # init optimizer
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr) # adam -> rmsprop
    print(f"device = {args.device}")

    ckpt_dir = args.ckpt_dir / "model_intent_2.pt"

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # Training loop - iterate over train dataloader and update model weights
        model.train()
        for batch in train_dataloader:
            inputs = torch.tensor(batch["text"], dtype=torch.int32).to(args.device)
            labels = torch.tensor(batch["label"], dtype=torch.int64).to(args.device)
            
            optimizer.zero_grad()
            outputs = model(inputs, batch["text_lengths"])
            #print(labels.shape)
            #print(outputs.shape)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(outputs)
        print(f"loss = {loss.item()}")

        # Evaluation loop - calculate accuracy and save model weights
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for batch in dev_dataloader:
                inputs = torch.tensor(batch["text"], dtype=torch.int32).to(args.device)
                labels = torch.tensor(batch["label"], dtype=torch.int64).to(args.device)

                predicts = model(inputs, batch["text_lengths"])
                loss = loss_fn(predicts, labels)

                for i in range(labels.size(0)): 
                    if torch.argmax(predicts[i]).item() == labels[i].item():
                        total_acc += 1
                total_count += labels.size(0)

        print(predicts)
        print(f"total_acc = {total_acc}")
        accuracy = float(total_acc)/total_count
        print(f"accuracy = {accuracy}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckpt_dir)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    ### change with batch size
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    ### mac os is use mps to GPU
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)

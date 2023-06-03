import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)  # len = 4117
    
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_dataloader = DataLoader(datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn, drop_last=True, shuffle=True)
    dev_dataloader = DataLoader(datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn, drop_last=True, shuffle=True)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")  # torch.Size([4117, 300])
    
    model = SeqTagger(embeddings, args.hidden_size, args.num_layers, args.dropout, args.bidirectional, len(tag2idx), args.device).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100).to(args.device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)  # adam -> rmsprop
    print(f"device = {args.device}")

    ckpt_dir = args.ckpt_dir / "model_slot_1.pt"

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        model.train()
        for batch in train_dataloader:
            inputs = torch.tensor(batch["text"], dtype=torch.int32).to(args.device)
            labels = torch.tensor(batch["label"], dtype=torch.int64).to(args.device)
            
            optimizer.zero_grad()

            outputs = model(inputs, batch["text_lengths"])
            outputs = outputs.permute(0,2,1)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
        
        #print(outputs)
        #print(f"loss = {loss.item()}")

        # Evaluation loop - calculate accuracy and save model weights
        model.eval()
        total_acc, total_count = 0, 0
        with torch.no_grad():
            for batch in dev_dataloader:
                inputs = torch.tensor(batch["text"], dtype=torch.int32).to(args.device)
                labels = torch.tensor(batch["label"], dtype=torch.int64).to(args.device)

                predicts = model(inputs, batch["text_lengths"])

                predicts = torch.argmax(predicts, dim=2)
                #loss = loss_fn(predicts, labels)

                for i in range(args.batch_size):
                    predict_list = []
                    label_list = []
                    for x in range(batch["text_lengths"][i]): 
                        predict_list.append(predicts[i][x].item())
                        label_list.append(labels[i][x].item())
                    if predict_list == label_list:
                        total_acc += 1
                total_count += labels.size(0)

        #print(predicts)
        #print(f"total_acc = {total_acc}")
        accuracy = float(total_acc)/total_count
        #print(f"accuracy = {accuracy}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, ckpt_dir)

    # Inference on test set
    true_list = []
    eval_list = []
    model.eval()
    with torch.no_grad():
        for batch in dev_dataloader:
            inputs = torch.tensor(batch["text"], dtype=torch.int32).to(args.device)
            labels = torch.tensor(batch["label"], dtype=torch.int64).to(args.device)

            predicts = model(inputs, batch["text_lengths"])
            predicts = predicts.argmax(dim=2)

            for i in range(predicts.size(0)):
                true_seq = []
                eval_seq = []
                for x in range(batch["text_lengths"][i]):
                    true_seq.append(dataset.idx2label(labels[i][x].item()))
                    eval_seq.append(dataset.idx2label(predicts[i][x].item()))
                true_list.append(true_seq)
                eval_list.append(eval_seq)

        print(classification_report(true_list, eval_list, mode='strict', scheme=IOB2))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=125)

    # training
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
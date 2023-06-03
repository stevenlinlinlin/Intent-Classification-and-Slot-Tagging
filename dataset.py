from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # implement collate_fn
        for i, dict in enumerate(samples):
            dict["unsort_id"] = i

        id_list = []
        unsort_id_list = []

        ## only for train and eval
        #label_list = []

        batch_sequence = []
        batch_text_lengths = []
        for data in sorted(samples, key=lambda x: len(x["text"].split(" ")), reverse=True):
            # text
            batch_sequence.append(data["text"].split(" "))
            batch_text_lengths.append(len(data["text"].split(" ")))

            # label
            ## only for train and eval
            #to_list = [0] * self.num_classes
            #to_list[self.label2idx(data["intent"])] = 1
            #label_list.append(self.label2idx(data["intent"]))

            # id
            id_list.append(data["id"])
            unsort_id_list.append(data["unsort_id"])
        
        text_list = self.vocab.encode_batch(batch_sequence)

        collate = {
            #"label": label_list,  ## only for train and eval
            "text": text_list,
            "id": id_list,
            "unsort_id": unsort_id_list,
            "text_lengths": batch_text_lengths
        }
        return collate

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # implement collate_fn
        for i, dict in enumerate(samples):
            dict["unsort_id"] = i

        id_list = []
        unsort_id_list = []

        ## only fot train and eval
        #label_list = []
        #label_seq_list = []
        
        batch_sequence = []
        batch_text_lengths = []

        for data in sorted(samples, key=lambda x: len(x["tokens"]), reverse=True):
            # token
            batch_sequence.append(data["tokens"])
            batch_text_lengths.append(len(data["tokens"]))

            # id
            unsort_id_list.append(data["unsort_id"])
            id_list.append(data["id"])

            # tag
            ## only for train and eval
            #to_list = []
            #for tag in data["tags"]:
                #to_list.append(self.label2idx(tag))
            #label_seq_list.append(to_list)      
        #label_list = pad_to_len(label_seq_list, batch_text_lengths[0], -100)

        text_list = self.vocab.encode_batch(batch_sequence)

        collate = {
            #"label": label_list,  ## only for train and eval
            "text": text_list,
            "id": id_list,
            "unsort_id": unsort_id_list,
            "text_lengths": batch_text_lengths
        }
        return collate

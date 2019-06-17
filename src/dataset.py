import torch
from torch.utils.data import Dataset, DataLoader


class PNDataset(Dataset):
    def __init__(self, datafile, word2id, max_len):
        self.w2i = word2id
        self.max_len: int = max_len
        self.sources, self.targets = self._load(datafile)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        mask = [1] * len(source)
        target = self.targets[idx]
        return source, mask, target

    def _s2id(self, sentence):
        idx = []
        sentence = sentence.strip('\n').strip().split()
        for word in sentence:
            word_id = self.w2i.get(word)
            if word_id is None:
                word_id = self.w2i.get('<UNK>')
            idx.append(word_id)
        return idx

    def _load(self, datafile):
        sources, targets = [], []
        with open(datafile, "r") as f:
            lines = f.readlines()
            self.data_size = len(lines)
            for i, line in enumerate(lines):
                target, sentence = line.strip().split('\t')
                if self.max_len > 0 and self.max_len <= len(sentence):
                    sentence = sentence[-self.max_len:]
                target = int(target)
                if target == -1:
                    target = 0
                sources.append(self._s2id(sentence))
                targets.append(target)

        return sources, targets


class PNDataLoader(DataLoader):
    def __init__(self,
                 datafile,
                 word2id,
                 max_len,
                 batch_size: int,
                 shuffle: bool,
                 num_workers: int):
        self.dataset = PNDataset(datafile, word2id, max_len)
        self.n_samples = len(self.dataset)
        super(PNDataLoader, self).__init__(self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           collate_fn=my_collate_fn)


def my_collate_fn(batches):
    sources, masks, targets = [], [], []
    max_len_in_batch = max(len(batch[0]) for batch in batches)
    for batch in batches:
        source, mask, target = batch
        pad = [1] * (max_len_in_batch - len(source))
        sources.append(source + pad)
        masks.append(mask + pad)
        targets.append(target)
    return torch.LongTensor(sources), torch.LongTensor(masks), torch.LongTensor(targets)

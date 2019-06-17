import argparse
import yaml
from attrdict import AttrDict

import torch
import torch.nn.functional as F

from dataset import PNDataLoader
from modeling import BiLSTM


def metric_fn(output, target):
    predict = torch.argmax(output, dim=1)
    return (predict == target).sum().item()


def loss_fn(output: torch.Tensor,
            target: torch.Tensor
            ) -> torch.Tensor:
    softmax = F.softmax(output, dim=1)
    loss = F.binary_cross_entropy(softmax[:, 1], target.float(), reduction='sum')
    return loss


def parser_init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="where is the config file")
    args = parser.parse_args()
    config = AttrDict(yaml.load(open(args.config, "r")))

    return config


def main():
    args = parser_init()
    print(args)

    word2id = {}
    with open(args.vocab_path, "r") as f:
        lines = f.readlines()
        dict_size = len(lines)
        for idx, line in enumerate(lines):
            word = line.strip()
            word2id[word] = idx

    test_dataset = PNDataLoader(args.test_path, word2id, args.max_len, args.batch_size,
                                shuffle=False, num_workers=2)
    model = BiLSTM(input_size=dict_size,
                   hidden_size=args.hidden_size,
                   num_layers=args.encoder_layer_num,
                   emb_size=args.emb_size,
                   dropout=args.dropout,device=args.device,
                   bidirectional=args.bidirectional)
    state_dict = torch.load(args.load_path, map_location=args.device)
    model.load_state_dict(state_dict)

    model.to(args.device)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        tp, tn, fn, fp = 0, 0, 0, 0
        for source, mask, target in test_dataset:
            source = source.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)
            output = model(source, mask)

            predict = torch.argmax(output, dim=1)
            for t, p in zip(target, predict):
                if t == 1 and p == 1:
                    tp += 1
                if t == 0 and p == 0:
                    tn += 1
                if t == 1 and p == 0:
                    fn += 1
                if t == 0 and p == 1:
                    fp += 1
            total_loss += loss_fn(output, target)
            total_correct += metric_fn(output, target)

    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    result_filename = args.save_path + ".txt"
    with open(result_filename, "w") as f:
        f.write(f'test_loss={total_loss / test_dataset.n_samples:.3f}\n')
        f.write(f'test_accuracy={total_correct / test_dataset.n_samples:.3f}\n')
        f.write("TP: {}\nTN: {}\nFP: {}\nFN: {}\nPrecision: {}\nRecall: {}\nF1: {}\n"
                .format(tp, tn, fp, fn, precision, recall, f1))
    print(f'test_loss={total_loss / test_dataset.n_samples:.3f}', end=' ')
    print(f'test_accuracy={total_correct / test_dataset.n_samples:.3f}')
    print("TP: {}\nTN: {}\nFP: {}\nFN: {}\nPrecision: {}\nRecall: {}\nF1: {}\n"
          .format(tp, tn, fp, fn, precision, recall, f1))


if __name__ == '__main__':
    main()

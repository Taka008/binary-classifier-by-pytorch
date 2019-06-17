import argparse
import yaml
from attrdict import AttrDict

import torch
import torch.optim as optim
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

    torch.manual_seed(args.seed)
    train_dataset = PNDataLoader(args.train_path, word2id, args.max_len, args.batch_size,
                                 shuffle=True, num_workers=2)
    dev_dataset = PNDataLoader(args.dev_path, word2id, args.max_len, args.batch_size,
                               shuffle=False, num_workers=2)

    model = BiLSTM(input_size=dict_size,
                hidden_size=args.hidden_size,
                num_layers=args.encoder_layer_num,
                emb_size=args.emb_size,
                dropout=args.dropout,
                device=args.device,
                bidirectional=args.bidirectional)

    model = model.to(args.device)
    print("model loaded: {}".format(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    step = 0
    dev_step = 0
    best_valid_acc = -1

    for epoch in range(1, args.epochs + 1):
        print(f'*** epoch {epoch} ***')
        model.train()
        total_loss = 0
        total_correct = 0
        for source, mask, target in train_dataset:
            step += 1
            source = source.to(args.device)
            mask = mask.to(args.device)
            target = target.to(args.device)
            output = model.forward(source, mask)
            # loss = criterion(output, target)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += metric_fn(output, target)

        print(f'train_loss={total_loss/train_dataset.n_samples:.3f}', end=' ')
        # print(f'train_loss={total_loss:.3f}', end=' ')
        print(f'train_accuracy={total_correct / train_dataset.n_samples:.3f}')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for source, mask, target in dev_dataset:
                dev_step += 1
                source = source.to(args.device)
                mask = mask.to(args.device)
                target = target.to(args.device)
                output = model.forward(source, mask)

                loss = loss_fn(output, target)
                # loss = criterion(output, target)
                total_loss += loss.item()
                total_correct += metric_fn(output, target)

            valid_acc = total_correct / dev_dataset.n_samples
            print(f'valid_loss={total_loss / dev_dataset.n_samples:.3f}', end=' ')
            # print(f'valid_loss={total_loss:.3f}', end=' ')
            print(f'valid_accuracy={valid_acc:.3f}')
            if valid_acc > best_valid_acc:
                torch.save(model.state_dict(), args.save_path)
                best_valid_acc = valid_acc


if __name__ == '__main__':
    main()

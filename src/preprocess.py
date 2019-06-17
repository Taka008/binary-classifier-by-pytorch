import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data-path", help="path to data directory")
parser.add_argument("--dict-path", help="path to save the dictionary")
parser.add_argument("--vocab-size", type=int, default=32000, help="the size of the vocabulary")
args = parser.parse_args()

# make dictionary
UNK = '<UNK>'
PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'


def generate_dict(data_path, dict_path, vocab_size):
    dict = {}
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip('\n').split('\t')[1].split()
            for word in line:
                dict[word] = dict.get(word, 0) + 1
    dict_list = [[k, v] for k, v in dict.items()]
    dict_list.sort(key=lambda x: -x[1])
    dict_list = [x[0] for x in dict_list]
    dict_list = [UNK, PAD, SOS, EOS] + dict_list[0:vocab_size]

    with open(dict_path, 'w') as f:
        for key in dict_list:
            f.write(key + '\n')


generate_dict(args.data_path, args.dict_path, args.vocab_size)

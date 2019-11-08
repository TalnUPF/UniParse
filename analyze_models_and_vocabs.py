import os
import subprocess
import logging
from uniparse import Vocabulary


def compute_vocab_size(vocab_file):
    vocab = Vocabulary()
    vocab.load(vocab_file)
    logging.info('Size of %s: %s' % (vocab_file, vocab.vocab_size))


def count_sentences(conll_file):
    """
    When -c option is used with grep, it counts the number of occurrences of the search string and outputs the same.
    """
    result = subprocess.run(['grep', '-c', '^$', conll_file], stdout=subprocess.PIPE)
    logging.info('File %s has %s sentences.' % (conll_file, result.stdout.decode('utf-8').replace('\n', '')))


def analyze_models_and_vocabs(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file == 'vocab.pkl':
                compute_vocab_size(file_path)
            elif file.endswith('.conll') or file.endswith('.conllu'):
                count_sentences(file_path)


if __name__ == '__main__':
    logging.basicConfig(filename='analyze_models_and_vocabs.log',
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:\t%(message)s")

    paths = ['/home/lpmayos/hd/code/UniParse/models',
             '/home/lpmayos/hd/code/UniParse/datasets',
             '/homedtic/lperez/UniParse/models',
             '/homedtic/lperez/datasets']
    for path in paths:
        analyze_models_and_vocabs(path)

import os
import sys
import argparse
import logging
from pathlib import Path
from nltk.parse.corenlp import CoreNLPDependencyParser


def generate_individual_conllu(input_dir, output_dir):
    """
    """

    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    already_generated = os.listdir(output_dir)

    for file in sorted(os.listdir(input_dir)):

        output_file_name = '%s.conllu' % file
        if output_file_name not in already_generated:  # don't parse again files already parsed!

            input_file = '%s/%s' % (input_dir, file)
            output_file = '%s/%s' % (output_dir, output_file_name)

            with open(input_file, "r", encoding='utf-8') as i_f, open(output_file, "w", encoding='utf-8') as o_f1:
                sent_id = 1
                nl = 1
                lines = i_f.readlines()
                for chunk in _chunks(lines, 500):

                    sentences = [a.split() for a in chunk]
                    sentences_parses = dep_parser.parse_sents(sentences)
                    for i, sentence_parse in enumerate(sentences_parses):
                        for parse in sentence_parse:
                            conll = parse.to_conll(style=10)

                            # write to output_file
                            o_f1.write("# sent_id = %s\n" % (nl))
                            o_f1.write("%s\n" % conll)
                            nl+=1

                            if nl % 1000 == 0:
                                logging.info("file: %s; sent_id = %s" % (output_file, nl))

            i_f.close()
            o_f1.close()

        else:
            logging.info('skipping %s; already parsed!' % output_file_name)


def combine_individual_conllu(input_dir, output_file, limit_to_files=None):
    """
    """
    num_sentence_global = 0
    with open(output_file, "w", encoding='utf-8') as o_f:
    
        for file in sorted(os.listdir(input_dir)):

            if limit_to_files is None or file in limit_to_files:

                input_file = '%s/%s' % (input_dir, file)
                logging.info("combining file: %s" % (input_file))

                with open(input_file, "r", encoding='utf-8') as i_f:
                
                    num_sentence_file = 0
                    for line in i_f:

                        # replace local sentence number with global sentence number
                        if line.startswith('# sent_id = '):
                            num_sentence_global += 1
                            num_sentence_file += 1
                            line = "# sent_id = %s (file: %s; sent_id = %s)\n" % (num_sentence_global, input_file, num_sentence_file)

                        o_f.write(line)

                i_f.close()
    o_f.close()


def create_1b_train_dev_test_splits(input_dir, heldout_dir, training_dir, dev_set):
    """
    """

    #  test
    output_file = '%s/conll/%s.conllu' % (input_dir, '1B_test')
    combine_individual_conllu(heldout_dir, output_file, limit_to_files=None)

    # dev
    output_file = '%s/conll/%s.conllu' % (input_dir, '1B_dev')
    combine_individual_conllu(training_dir, output_file, limit_to_files=dev_set)

    # train
    train_set = [file for file in sorted(os.listdir(training_dir)) if file not in dev_set]
    output_file = '%s/conll/%s.conllu' % (input_dir, '1B_train')
    combine_individual_conllu(training_dir, output_file, limit_to_files=train_set)


if __name__ == "__main__":
    """ Usage:
    1) start CoreNLP Server:
        ~/code/stanford-corenlp-full-2018-02-27$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
    2) 1b_to_conll:
        dummy: python 1b_to_conll.py --input_dir ~/code/datasets/test_1B_dataset/ --heldout_folder heldout-monolingual.tokenized.shuffled --training_folder training-monolingual.tokenized.shuffled
        real:  python 1b_to_conll.py --input_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/ --heldout_folder heldout-monolingual.tokenized.shuffled --training_folder training-monolingual.tokenized.shuffled
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s:\t%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path, help="Path to folder containing the heldout and training folders of the 1B benchmark.")
    parser.add_argument("--heldout_folder", required=True, help="Name of the folder containing the heldout files of the 1B benchmark.")
    parser.add_argument("--training_folder", required=True, help="Name of the folder containing the training files of the 1B benchmark.")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    heldout_dir = '%s/text/%s' % (input_dir, args.heldout_folder)
    training_dir = '%s/text/%s' % (input_dir, args.training_folder)

    if not os.path.exists(heldout_dir):
        raise Exception(heldout_dir, " does not exist!")
    if not os.path.exists(training_dir):
        raise Exception(training_dir, " does not exist!")

    heldout_dir_conll = '%s/%s/%s' % (input_dir, 'conll_individual_files', args.heldout_folder)
    generate_individual_conllu(heldout_dir, heldout_dir_conll)

    training_dir_conll = '%s/%s/%s' % (input_dir, 'conll_individual_files', args.training_folder)
    generate_individual_conllu(training_dir, training_dir_conll)

    dev_set = ['news.en-000%s-of-00100.conllu' % a for a in range(80, 100)]  # for real
    # dev_set = ['news.en-00001-of-00100.conllu', 'news.en-00002-of-00100.conllu']  # for dummy
    create_1b_train_dev_test_splits(input_dir, heldout_dir_conll, training_dir_conll, dev_set)

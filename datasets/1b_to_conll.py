import os
import sys
import argparse
import logging
from pathlib import Path
from nltk.parse.corenlp import CoreNLPDependencyParser


def parse_file_to_conllu(input_file, output_file, num_sentence, all_sentences_file):

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    with open(input_file, "r", encoding='utf-8') as i_f, open(output_file, "w", encoding='utf-8') as o_f1, open(all_sentences_file, "a", encoding='utf-8') as o_f2:
        sent_id = 1
        nl = 1
        for line in i_f:
            parses = dep_parser.parse(line.split())

            for parse in parses:

                if nl % 10 == 0:
                    logging.info("file: %s; sent_id = %s" % (output_file, nl))

                # write to all_sentences_file
                o_f2.write("# sent_id = %s (file: %s; sent_id = %s)\n" % (num_sentence, output_file, nl))
                o_f2.write("# text = %s\n" % line.replace('\n', ''))
                o_f2.write("%s\n" % parse.to_conll(style=10))
                num_sentence += 1

                # write to output_file
                o_f1.write("# sent_id = %s\n" % (nl))
                o_f1.write("# text = %s\n" % line.replace('\n', ''))
                o_f1.write("%s\n" % parse.to_conll(style=10))
                nl+=1

    i_f.close()
    o_f1.close()
    o_f2.close()

    return num_sentence


if __name__ == "__main__":
    """ Usage:
    1) start CoreNLP Server:
        ~/code/stanford-corenlp-full-2018-02-27$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
    2) 1b_to_conll:
        dummy: python 1b_to_conll.py --input_dir ~/code/datasets/test_1B_dataset/heldout-monolingual.tokenized.shuffled/ --output_dir ~/code/datasets/test_1B_dataset/conll/heldout-monolingual.tokenized.shuffled/
        real:  python 1b_to_conll.py --input_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/ --output_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll/heldout-monolingual.tokenized.shuffled/
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s:\t%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path, help="Path to folder containing the files with sentences to parse.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Path to output folder where conllu files will be stored.")
    args = parser.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    all_sentences_file = '%s/%s.conllu' % (output_dir, 'all_sentences')

    if not os.path.exists(input_dir):
        raise Exception(input_dir, " does not exist!")
    if not os.path.exists(output_dir):
        raise Exception(output_dir, " does not exist!")

    num_sentence = 1
    for file in os.listdir(input_dir):
        in_file = '%s/%s' % (input_dir, file)
        out_file = '%s/%s.conllu' % (output_dir, file)
        num_sentence = parse_file_to_conllu(in_file, out_file, num_sentence, all_sentences_file)

import os
import sys
import argparse
import logging
from pathlib import Path
from nltk.parse.corenlp import CoreNLPDependencyParser


# def parse_files_to_conllu_and_main_conllu(input_dir, output_dir, all_sentences_file):

#     def _chunks(l, n):
#         """Yield successive n-sized chunks from l."""
#         for i in range(0, len(l), n):
#             yield l[i:i + n]

#     dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

#     num_sentence = 1
#     for file in sorted(os.listdir(input_dir)):
#         input_file = '%s/%s' % (input_dir, file)
#         output_file = '%s/%s.conllu' % (output_dir, file)

#         with open(input_file, "r", encoding='utf-8') as i_f, open(output_file, "w", encoding='utf-8') as o_f1, open(all_sentences_file, "a", encoding='utf-8') as o_f2:
#             sent_id = 1
#             nl = 1
#             lines = i_f.readlines()
#             for chunk in _chunks(lines, 500):

#                 sentences = [a.split() for a in chunk]
#                 sentences_parses = dep_parser.parse_sents(sentences)
#                 for i, sentence_parse in enumerate(sentences_parses):
#                     for parse in sentence_parse:
#                         conll = parse.to_conll(style=10)

#                         # write to all_sentences_file
#                         o_f2.write("# sent_id = %s (file: %s; sent_id = %s)\n" % (num_sentence, output_file, nl))
#                         o_f2.write("%s\n" % conll)
#                         num_sentence += 1

#                         # write to output_file
#                         o_f1.write("# sent_id = %s\n" % (nl))
#                         o_f1.write("%s\n" % conll)
#                         nl+=1

#                         if nl % 100 == 0:
#                             logging.info("file: %s; sent_id = %s" % (output_file, nl))

#         i_f.close()
#         o_f1.close()
#         o_f2.close()


def generate_individual_conllu(input_dir, output_dir):
    """
    """

    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

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


def combine_individual_conllu(input_dir, all_sentences_file):
    """
    """
    num_sentence_global = 0
    with open(all_sentences_file, "w", encoding='utf-8') as o_f:
    
        for file in sorted(os.listdir(input_dir)):

            logging.info("combining file: %s" % (file))

            input_file = '%s/%s' % (input_dir, file)
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


if __name__ == "__main__":
    """ Usage:
    1) start CoreNLP Server:
        ~/code/stanford-corenlp-full-2018-02-27$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
    2) 1b_to_conll:
        dummy: python 1b_to_conll.py --input_dir ~/code/datasets/test_1B_dataset/heldout-monolingual.tokenized.shuffled/ --output_dir ~/code/datasets/test_1B_dataset/conll/heldout-monolingual.tokenized.shuffled/
        real (heldout):  python 1b_to_conll.py --input_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/ --output_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll/heldout-monolingual.tokenized.shuffled/
        real (training):  python 1b_to_conll.py --input_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/ --output_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll/training-monolingual.tokenized.shuffled/
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

    # parse_files_to_conllu_and_main_conllu(input_dir, output_dir, all_sentences_file)

    generate_individual_conllu(input_dir, output_dir)
    combine_individual_conllu(output_dir, all_sentences_file)
import os
import sys
import argparse
import logging
from pathlib import Path
from nltk.parse.corenlp import CoreNLPDependencyParser


def prova1(sentences):
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    sentences = [a.split() for a in sentences]
    sentences_parses = dep_parser.parse_sents(sentences)
    for sentence_parse in sentences_parses:
        for parse in sentence_parse:
            print(parse.to_conll(style=10))


def prova2(sentences):
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    sentences = [a.split() for a in sentences]
    for sentence in sentences:
        sentence_parses = dep_parser.parse(sentence)
        for parse in sentence_parses:
            print(parse.to_conll(style=10))

def prova():
    """
    900 sentences
        prova1 (parse_sents): 8.950937271118164
        prova2 (parse): 52.62314772605896

    --> 30301028 total sentences --> 

    """
    import time

    sentences = ["Having a little flexibility on that issue would go a long way to putting together a final package .", "Long before the advent of e-commerce , Wal-Mart 's founder Sam Walton set out his vision for a successful retail operation : We let folks know we 're interested in them and that they 're vital to us-- ' cause they are , he said .", "A spokesman said the company has been affected by the credit crunch in the United States .", "Abu Dhabi is going ahead to build solar city and no pollution city .", "Her back was torn open , her liver was ruptured , one of her lungs had collapsed and the other was punctured .", "Now it has 175 staging centers , where volunteers are coordinating get-out-the-vote efforts , said Obama 's Georgia spokeswoman , Caroline Adelman .", "How about a sibling or family friend ?", "Butler 's the scorer .", "In the meantime , the multi-talented Bell gets to showcase her musical chops during the end credits of When in Rome when the cast breaks in to a musical dance number .", "McCain said he must convince Americans that protectionism and isolationism could be harmful ."]
    sentences = sentences * 90


    start = time.time()
    prova1(sentences)
    end = time.time()
    time1 = end - start

    start = time.time()
    prova2(sentences)
    end = time.time()
    time2 = end - start

    print('%s sentences' % len(sentences))
    print('prova1 (parse_sents): %s' % time1)
    print('prova2 (parse): %s' % time2)




def parse_file_to_conllu(input_file, output_file, num_sentence, all_sentences_file):

    def _chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

    with open(input_file, "r", encoding='utf-8') as i_f, open(output_file, "w", encoding='utf-8') as o_f1, open(all_sentences_file, "a", encoding='utf-8') as o_f2:
        sent_id = 1
        nl = 1
        lines = i_f.readlines()
        for chunk in _chunks(lines, 500):

            sentences = [a.split() for a in chunk]
            sentences_parses = dep_parser.parse_sents(sentences)
            for i, sentence_parse in enumerate(sentences_parses):
                for parse in sentence_parse:
                    conll = parse.to_conll(style=10)

                    # write to all_sentences_file
                    o_f2.write("# sent_id = %s (file: %s; sent_id = %s)\n" % (num_sentence, output_file, nl))
                    o_f2.write("%s\n" % conll)
                    num_sentence += 1

                    # write to output_file
                    o_f1.write("# sent_id = %s\n" % (nl))
                    o_f1.write("%s\n" % conll)
                    nl+=1

                    if nl % 100 == 0:
                        logging.info("file: %s; sent_id = %s" % (output_file, nl))

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
    for file in sorted(os.listdir(input_dir)):
        in_file = '%s/%s' % (input_dir, file)
        out_file = '%s/%s.conllu' % (output_dir, file)
        num_sentence = parse_file_to_conllu(in_file, out_file, num_sentence, all_sentences_file)

    # prova()

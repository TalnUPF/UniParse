import argparse
import logging
import os

from kiperwasser_main import transform_to_conllu
from uniparse import Vocabulary, Model
from uniparse.callbacks import TensorboardLoggerCallback, ModelSaveCallback
from uniparse.models.kiperwasser import DependencyParser


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", dest="input_file", help="File that we want to extract embeddings for", metavar="FILE", required=True)
    parser.add_argument("--vocab_file", dest="vocab_file", required=True)
    parser.add_argument("--model_file", dest="model_file", required=True)
    parser.add_argument("--logging_file", dest="logging_file", help="File to store the logs", metavar="FILE", required=True)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)

    arguments, unknown = parser.parse_known_args()

    # configure logging

    logging.basicConfig(filename=arguments.logging_file,
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:\t%(message)s")

    logging.info("\n\n\n===================================================================================================")
    logging.info("kiperwasser_embeddings_extractor")
    logging.info("===================================================================================================\n")

    logging.info("\nArguments:")
    logging.info(arguments)
    logging.info("\n")

    # load vocabulary

    only_words = True
    vocab = Vocabulary(only_words)
    vocab.load(arguments.vocab_file)


    # load model

    embs = None
    model = DependencyParser(vocab, embs, False)
    parser = Model(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    parser.load_from_file(arguments.model_file)


    # transform input files into conllu if needed

    arguments.input_file = transform_to_conllu(arguments.input_file)


    # parse test file

    input_data = vocab.tokenize_conll(arguments.input_file)
    embeddings = parser.extract_embeddings(input_data, arguments.batch_size)
    first_embedding = embeddings[0][0]
    output = first_embedding.h()
    hidden_state = first_embedding.s()
    s_f = hidden_state[0]
    s_b = hidden_state[1]
    print(s_f.value())
    print(s_b.value())


if __name__ == '__main__':
    main()

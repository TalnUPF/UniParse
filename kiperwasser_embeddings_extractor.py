import argparse
import logging

from kiperwasser_main import transform_to_conllu
from uniparse import Vocabulary, ParserModel
from uniparse.models.kiperwasser import DependencyParser as DependencyParser
from uniparse.models.pytorch_kiperwasser import DependencyParser as DependencyParserPytorch


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--input_file", dest="input_file", help="File that we want to extract embeddings for", metavar="FILE", required=False)
    arg_parser.add_argument("--vocab_file", dest="vocab_file", required=True)
    arg_parser.add_argument("--model_file", dest="model_file", required=True)
    arg_parser.add_argument("--logging_file", dest="logging_file", help="File to store the logs", metavar="FILE", required=True)
    arg_parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)

    arguments, unknown = arg_parser.parse_known_args()

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

    backend = 'pytorch'
    if backend == 'dynet':
        embs = None
        parser = DependencyParser(vocab, embs, False)
    elif backend == 'pytorch':
        arguments.vocab_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/run2/vocab.pkl'
        arguments.model_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/run2/model.model'
        only_words = False
        vocab = Vocabulary(only_words)
        vocab.load(arguments.vocab_file)
        parser = DependencyParserPytorch(vocab)

    model = ParserModel(parser, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    model.load_from_file(arguments.model_file)

    if arguments.input_file is not None:
        arguments.input_file = transform_to_conllu(arguments.input_file)
        input_data = vocab.tokenize_conll(arguments.input_file)

    else:
        words = ('Chancellor', 'of', 'the', 'Exchequer', 'Nigel', 'Lawson', "'s", 'restated', 'commitment', 'to', 'a', 'firm', 'monetary', 'policy', 'has', 'helped', 'to', 'prevent', 'a', 'freefall', 'in', 'sterling', 'over', 'the', 'past', 'week', '.')
        tags = ('O', 'B-PP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'I-NP', 'B-PP', 'B-NP', 'I-NP', 'I-NP', 'I-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'B-PP', 'B-NP', 'I-NP', 'I-NP', 'O')
        input_data = vocab.word_tags_tuple_to_conll(words, tags)

    embeddings = model.extract_embeddings(input_data)

    print(embeddings)

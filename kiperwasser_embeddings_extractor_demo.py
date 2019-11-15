import logging

from kiperwasser_main import transform_to_conllu
from uniparse import Vocabulary, ParserModel
from uniparse.models.kiperwasser import DependencyParser as DependencyParser
from uniparse.models.pytorch_kiperwasser import DependencyParser as DependencyParserPytorch


if __name__ == '__main__':

    # configure logging

    logging_file = '/home/lpmayos/hd/code/UniParse/logging.log'
    logging.basicConfig(filename=logging_file,
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:\t%(message)s")

    logging.info("\n\n\n===================================================================================================")
    logging.info("kiperwasser_embeddings_extractor")
    logging.info("===================================================================================================\n")

    # load model and  vocab

    backend = 'pytorch'
    # backend = 'dynet'

    if backend == 'dynet':
        vocab_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser/1b/bpe/mini/only_words_true/run1/vocab.pkl'
        model_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser/1b/bpe/mini/only_words_true/run1/model.model'
        only_words = True
        vocab = Vocabulary(only_words)
        vocab.load(vocab_file)
        embs = None
        parser = DependencyParser(vocab, embs, False)
    elif backend == 'pytorch':

        run = 1

        if run == 1:
            vocab_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/vocab.pkl'
            model_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/model.model'
            upos_dim = 25
            word_dim = 100
            hidden_dim = 100
        elif run == 2:
            vocab_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run2/vocab.pkl'
            model_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run2/model.model'
            upos_dim = 25
            word_dim = 100
            hidden_dim = 200

        only_words = False
        vocab = Vocabulary(only_words)
        vocab.load(vocab_file)
        parser = DependencyParserPytorch(vocab, upos_dim, word_dim, hidden_dim)

    model = ParserModel(parser, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    model.load_from_file(model_file)

    # input_file = '/home/lpmayos/hd/code/cvt_text/data/raw_data/depparse/test_mini.txt'
    # input_file = None
    input_file = '/home/lpmayos/hd/code/structural-probes/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu'

    if input_file is not None:
        input_file = transform_to_conllu(input_file)
        input_data = vocab.tokenize_conll(input_file)

    else:
        words = ('Chancellor', 'of', 'the', 'Exchequer', 'Nigel', 'Lawson', "'s", 'restated', 'commitment', 'to', 'a', 'firm', 'monetary', 'policy', 'has', 'helped', 'to', 'prevent', 'a', 'freefall', 'in', 'sterling', 'over', 'the', 'past', 'week', '.')
        tags = ('O', 'B-PP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'I-NP', 'B-PP', 'B-NP', 'I-NP', 'I-NP', 'I-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'B-PP', 'B-NP', 'I-NP', 'I-NP', 'O')
        input_data = vocab.word_tags_tuple_to_conll(words, tags)

    embeddings = model.extract_embeddings(input_data)

    print(embeddings)

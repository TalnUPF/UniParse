import logging

from kiperwasser_main import transform_to_conllu
from uniparse import Vocabulary, ParserModel
from uniparse.models.kiperwasser import DependencyParser as DependencyParser


""" lpmayos NOTE
This demo was actually never used, as we switched to use the pytorch backend.
This means that the way to retrieve the embeddings concatenated may not be useful/usable.

For an updated version, please check kiperwasser_contextual_embeddings_pytorch_demo.py 
"""
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

    vocab_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser/1b/bpe/mini/only_words_true/run1/vocab.pkl'
    model_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser/1b/bpe/mini/only_words_true/run1/model.model'
    only_words = True
    vocab = Vocabulary(only_words)
    vocab.load(vocab_file)
    embs = None
    parser = DependencyParser(vocab, embs, False)

    model = ParserModel(parser, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    model.load_from_file(model_file)

    # input_file = '/home/lpmayos/hd/code/cvt_text/data/raw_data/depparse/test_mini.txt'
    input_file = '/home/lpmayos/hd/code/structural-probes/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu'

    input_file = transform_to_conllu(input_file)
    input_data = vocab.tokenize_conll(input_file)

    embeddings = parser.extract_embeddings(input_data, model.backend, format='concat', save=True, file_path='babau.hdf5')  # {'0': <np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>, '1':<np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>...}

    print(embeddings)

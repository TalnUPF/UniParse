import collections
import copy
import logging

import h5py
import torch

from kiperwasser_main import transform_to_conllu
from uniparse import Vocabulary, ParserModel
from uniparse.models.kiperwasser_pytorch import DependencyParser as DependencyParserPytorch
import torch.nn as nn
import numpy as np

""" 
We decouple the model to extract and use embeddings, generating separate LSTM for each layer and direction.

- TODO make sure that a K&G parser with those decoupled LSTM raises the same results as one trained with original biLSTM.  
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

    # load original K&G model and  vocab

    vocab_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/vocab.pkl'
    only_words = False
    vocab = Vocabulary(only_words)
    vocab.load(vocab_file)
    model_file = '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/model.model'
    upos_dim = 25
    word_dim = 100
    hidden_dim = 100
    parser = DependencyParserPytorch(vocab, upos_dim, word_dim, hidden_dim)
    model = ParserModel(parser, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    model.load_from_file(model_file)

    # generate input data

    # input_file = '/home/lpmayos/hd/code/cvt_text/data/raw_data/depparse/test_mini.txt'
    input_file = '/home/lpmayos/hd/code/structural-probes/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu'

    input_file = transform_to_conllu(input_file)
    input_data = vocab.tokenize_conll(input_file)

    # create and initialize FWD and BWD biLSTMs with model parameters

    input_size = word_dim + upos_dim

    named_parameters = {key: value for (key, value) in parser.deep_bilstm.named_parameters()}

    state_dict = parser.deep_bilstm.state_dict()

    lstm_fwd_0 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
    new_state_dict = collections.OrderedDict()
    new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l0']
    new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l0']
    new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l0']
    new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l0']
    lstm_fwd_0.load_state_dict(new_state_dict)

    lstm_bwd_0 = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
    new_state_dict = collections.OrderedDict()
    new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l0_reverse']
    new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l0_reverse']
    new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l0_reverse']
    new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l0_reverse']
    lstm_bwd_0.load_state_dict(new_state_dict)

    # NOTICE! input_size = 2*hidden_dim?
    lstm_fwd_1 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
    new_state_dict = collections.OrderedDict()
    new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l1']
    new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l1']
    new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l1']
    new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l1']
    lstm_fwd_1.load_state_dict(new_state_dict)

    # NOTICE! input_size = 2*hidden_dim?
    lstm_bwd_1 = nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
    new_state_dict = collections.OrderedDict()
    new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l1_reverse']
    new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l1_reverse']
    new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l1_reverse']
    new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l1_reverse']
    lstm_bwd_1.load_state_dict(new_state_dict)

    # process data

    embeddings = {}
    for i, sample in enumerate(input_data):
        model.backend.renew_cg()  # for pytorch it is just 'pass'

        # get embeddings

        words, lemmas, tags, heads, rels, chars = sample

        words = model.backend.input_tensor(np.array([words]), dtype="int")
        tags = model.backend.input_tensor(np.array([tags]), dtype="int")

        word_embs = parser.wlookup(words)
        tags_embs = parser.tlookup(tags)

        input_data = torch.cat([word_embs, tags_embs], dim=-1)  # dim 1x8x125 (if we have 8 words in the sentence)
        input_data_reversed = torch.flip(input_data, (1,))

        # feed data

        out_lstm_fwd_0, hidden_lstm_fwd_0 = lstm_fwd_0(input_data)
        out_lstm_bwd_0, hidden_lstm_bwd_0 = lstm_bwd_0(input_data_reversed)

        input_data = torch.cat((out_lstm_fwd_0, out_lstm_bwd_0), 2)
        out_lstm_fwd_1, hidden_lstm_fwd_1 = lstm_fwd_1(input_data)
        out_lstm_bwd_1, hidden_lstm_bwd_1 = lstm_bwd_1(input_data)

        # generate embeddings
        # TODO in ELMo they perform a task-dependant weighted sum of the concatenation of L0 (initial embeddings), L1 and L2
        # For now we'll sum there without weighting just for testing purposes

        sentence_embeddings = out_lstm_fwd_0 + out_lstm_bwd_0 + out_lstm_fwd_1 + out_lstm_bwd_1
        embeddings[i] = sentence_embeddings

    # save embeddings in hdf5 format

    # Write contextual word representations to disk for each of the train, dev, and test split in hdf5 format, where the
    # index of the sentence in the conllx file is the key to the hdf5 dataset object. That is, your dataset file should
    # look a bit like {'0': <np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>, '1':<np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>...}, etc.
    # Note here that SEQLEN for each sentence must be the number of tokens in the sentence as specified by the conllx file.

    with h5py.File('myfile.hdf5', 'w') as f:
        for k, v in embeddings.items():
            f.create_dataset(str(k), data=v.detach().numpy())

    # with h5py.File('myfile.hdf5', 'r') as f:
    #     data = f['0']
    #     print(data)

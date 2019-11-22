import collections
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


class EmbeddingsExtractor(object):

    def __init__(self, logging_file, model_config):
        self.logging_file = logging_file
        self.model_config = model_config

        # configure logging
        self._configure_logging()

        # load vocabilary, parser and model
        self._load_model()
        self._create_lstms()

    def _configure_logging(self):
        logging.basicConfig(filename=self.logging_file,
                            level=logging.DEBUG,
                            format="%(asctime)s:%(levelname)s:\t%(message)s")

    def _load_model(self):
        """ load original K&G model and  vocab
        """
        self.vocab = Vocabulary(self.model_config['only_words'])
        self.vocab.load(self.model_config['vocab_file'])
        self.parser = DependencyParserPytorch(self.vocab, self.model_config['upos_dim'], self.model_config['word_dim'], self.model_config['hidden_dim'])
        self.model = ParserModel(self.parser, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=self.vocab)
        self.model.load_from_file(self.model_config['model_file'])

    def _create_lstms(self):
        # create and initialize FWD and BWD biLSTMs with model parameters

        input_size = self.model_config['word_dim'] + self.model_config['upos_dim']

        state_dict = self.parser.deep_bilstm.state_dict()

        self.lstm_fwd_0 = nn.LSTM(input_size=input_size, hidden_size=self.model_config['hidden_dim'], num_layers=1, batch_first=True, bidirectional=False)
        new_state_dict = collections.OrderedDict()
        new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l0']
        new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l0']
        new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l0']
        new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l0']
        self.lstm_fwd_0.load_state_dict(new_state_dict)

        self.lstm_bwd_0 = nn.LSTM(input_size=input_size, hidden_size=self.model_config['hidden_dim'], num_layers=1, batch_first=True, bidirectional=False)
        new_state_dict = collections.OrderedDict()
        new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l0_reverse']
        new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l0_reverse']
        new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l0_reverse']
        new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l0_reverse']
        self.lstm_bwd_0.load_state_dict(new_state_dict)

        # NOTICE! input_size = 2*hidden_dim?
        self.lstm_fwd_1 = nn.LSTM(input_size=2*self.model_config['hidden_dim'], hidden_size=self.model_config['hidden_dim'], num_layers=1, batch_first=True, bidirectional=False)
        new_state_dict = collections.OrderedDict()
        new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l1']
        new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l1']
        new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l1']
        new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l1']
        self.lstm_fwd_1.load_state_dict(new_state_dict)

        # NOTICE! input_size = 2*hidden_dim?
        self.lstm_bwd_1 = nn.LSTM(input_size=2*self.model_config['hidden_dim'], hidden_size=self.model_config['hidden_dim'], num_layers=1, batch_first=True, bidirectional=False)
        new_state_dict = collections.OrderedDict()
        new_state_dict['weight_hh_l0'] = state_dict['lstm.weight_hh_l1_reverse']
        new_state_dict['weight_ih_l0'] = state_dict['lstm.weight_ih_l1_reverse']
        new_state_dict['bias_hh_l0'] = state_dict['lstm.bias_hh_l1_reverse']
        new_state_dict['bias_ih_l0'] = state_dict['lstm.bias_ih_l1_reverse']
        self.lstm_bwd_1.load_state_dict(new_state_dict)

    def generate_embeddings(self, input_file):

        logging.info("\n\n\n===================================================================================================")
        logging.info("kiperwasser_contextual embeddings_extractor")
        logging.info("===================================================================================================\n")

        # generate tokenized data
        input_data = self.vocab.tokenize_conll(input_file)

        embs = {}
        for i, sample in enumerate(input_data):
            self.model.backend.renew_cg()  # for pytorch it is just 'pass'

            # get embeddings

            words, lemmas, tags, heads, rels, chars = sample

            words = self.model.backend.input_tensor(np.array([words]), dtype="int")
            tags = self.model.backend.input_tensor(np.array([tags]), dtype="int")

            word_embs = self.parser.wlookup(words)
            tags_embs = self.parser.tlookup(tags)

            input_data = torch.cat([word_embs, tags_embs], dim=-1)  # dim 1x8x125 (if we have 8 words in the sentence)
            input_data_reversed = torch.flip(input_data, (1,))

            # feed data

            out_lstm_fwd_0, hidden_lstm_fwd_0 = self.lstm_fwd_0(input_data)
            out_lstm_bwd_0, hidden_lstm_bwd_0 = self.lstm_bwd_0(input_data_reversed)

            input_data = torch.cat((out_lstm_fwd_0, out_lstm_bwd_0), 2)
            input_data_reversed = torch.flip(input_data, (1,))
            out_lstm_fwd_1, hidden_lstm_fwd_1 = self.lstm_fwd_1(input_data)
            out_lstm_bwd_1, hidden_lstm_bwd_1 = self.lstm_bwd_1(input_data_reversed)

            # generate embeddings
            # TODO in ELMo they perform a task-dependant weighted sum of the concatenation of L0 (initial embeddings), L1 and L2
            # For now we'll sum there without weighting just for testing purposes

            sentence_embeddings = out_lstm_fwd_0 + out_lstm_bwd_0 + out_lstm_fwd_1 + out_lstm_bwd_1
            embs[i] = sentence_embeddings

        return embs

    def save_to_hdf5(self, embeddings, file_path):
        # save embeddings in hdf5 format

        # Write contextual word representations to disk for each of the train, dev, and test split in hdf5 format, where the
        # index of the sentence in the conllx file is the key to the hdf5 dataset object. That is, your dataset file should
        # look a bit like {'0': <np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>, '1':<np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>...}, etc.
        # Note here that SEQLEN for each sentence must be the number of tokens in the sentence as specified by the conllx file.

        with h5py.File(file_path, 'w') as f:
            for k, v in embeddings.items():
                f.create_dataset(str(k), data=v.detach().numpy())

    def check_hdf5_file(self, file_path):

        with h5py.File('myfile.hdf5', 'r') as f:
            for item in f.items():
                print(item)


if __name__ == '__main__':

    logging_file = '/home/lpmayos/hd/code/UniParse/logging.log'

    model_config = {
        'vocab_file': '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/vocab.pkl',
        'model_file': '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/model.model',
        'only_words': False,
        'upos_dim': 25,
        'word_dim': 100,
        'hidden_dim': 100
    }

    embeddings_extractor = EmbeddingsExtractor(logging_file, model_config)

    # generate input data

    # input_file = '/home/lpmayos/hd/code/cvt_text/data/raw_data/depparse/test_mini.txt'
    input_file = '/home/lpmayos/hd/code/structural-probes/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu'
    input_file = transform_to_conllu(input_file)

    # compute embeddings

    embs = embeddings_extractor.generate_embeddings(input_file)

    # save embedidngs in hdf5 format

    output_file = 'myfile.hdf5'
    embeddings_extractor.save_to_hdf5(embs, output_file)
    embeddings_extractor.check_hdf5_file(output_file)

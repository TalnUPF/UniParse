import collections
import logging
import os

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

        # configure logging
        self.logging_file = logging_file
        self._configure_logging()

        self.model_config = model_config
        logging.info(model_config)

        # load vocabulary, parser and model
        self._load_model()

        # create lstms
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
        logging.info("Generating K&G contextual embeddings for %s" % input_file)
        logging.info("===================================================================================================\n")

        # generate tokenized data
        tokenized_sentences = self.vocab.tokenize_conll(input_file)

        embs = {}
        for i, sample in enumerate(tokenized_sentences):
            self.model.backend.renew_cg()  # for pytorch it is just 'pass'

            # get embeddings

            words, lemmas, tags, heads, rels, chars = sample

            words = self.model.backend.input_tensor(np.array([words]), dtype="int")
            tags = self.model.backend.input_tensor(np.array([tags]), dtype="int")

            word_embs = self.parser.wlookup(words)
            tags_embs = self.parser.tlookup(tags)  # TODO think if it makes sense to use tag_embs or not!

            input_data0 = torch.cat([word_embs, tags_embs], dim=-1)  # dim 1x8x125 (if we have 8 words in the sentence)
            input_data0_reversed = torch.flip(input_data0, (1,))

            # feed data

            out_lstm_fwd_0, hidden_lstm_fwd_0 = self.lstm_fwd_0(input_data0)
            out_lstm_bwd_0, hidden_lstm_bwd_0 = self.lstm_bwd_0(input_data0_reversed)

            input_data1 = torch.cat((out_lstm_fwd_0, out_lstm_bwd_0), 2)
            input_data1_reversed = torch.flip(input_data1, (1,))
            out_lstm_fwd_1, hidden_lstm_fwd_1 = self.lstm_fwd_1(input_data1)
            out_lstm_bwd_1, hidden_lstm_bwd_1 = self.lstm_bwd_1(input_data1_reversed)

            # generate embeddings

            out_lstm_bwd_0 = torch.flip(out_lstm_bwd_0, (1,))
            out_lstm_bwd_1 = torch.flip(out_lstm_bwd_1, (1,))

            # TODO in ELMo they perform a task-dependant weighted sum of the concatenation of L0 (initial embeddings), L1 and L2;
            #  As our input has varying sizes and we are not weighting the layers, we'll just concatenate everything.
            # TODO for the syntactic probes, ELMo stores sepparately the three layers, so maybe we can do the same at least with layer 0 and layer1 ¿?
            sentence_embeddings = torch.cat((input_data0, out_lstm_fwd_0, out_lstm_bwd_0, out_lstm_fwd_1, out_lstm_bwd_1), 2)  # 1 x 8 x 125+100+100+100+100 = 525
            embs[i] = sentence_embeddings

        return embs

    @staticmethod
    def save_to_hdf5(embeddings, file_path, skip_root=False):
        # save embeddings in hdf5 format

        # Write contextual word representations to disk for each of the train, dev, and test split in hdf5 format, where the
        # index of the sentence in the conllx file is the key to the hdf5 dataset object. That is, your dataset file should
        # look a bit like {'0': <np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>, '1':<np.ndarray(size=(1,SEQLEN1,FEATURE_COUNT))>...}, etc.
        # Note here that SEQLEN for each sentence must be the number of tokens in the sentence as specified by the conllx file.

        with h5py.File(file_path, 'w') as f:
            for k, v in embeddings.items():
                logging.info('creating dataset for k %s' % str(k))
                sentence_embs = v.detach().numpy()
                if skip_root:
                    sentence_embs = sentence_embs[:, 1:, :]
                f.create_dataset(str(k), data=sentence_embs)

    @staticmethod
    def check_hdf5_file(file_path):

        with h5py.File(file_path, 'r') as f:
            for item in f.items():
                logging.info(item)


if __name__ == '__main__':

    input_files = ['/home/lpmayos/hd/code/structural-probes/lpmayos_tests/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu',
                   '/home/lpmayos/hd/code/structural-probes/lpmayos_tests/data/en_ewt-ud-sample/en_ewt-ud-test.conllu',
                   '/home/lpmayos/hd/code/structural-probes/lpmayos_tests/data/en_ewt-ud-sample/en_ewt-ud-train.conllu']

    model_config = {
        'vocab_file': '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/vocab.pkl',
        'model_file': '/home/lpmayos/hd/code/UniParse/models/kiperwasser_pytorch/ud/only_words_false/toy_runs/run1/model.model',
        'only_words': False,
        'upos_dim': 25,
        'word_dim': 100,
        'hidden_dim': 100
    }

    output_folder = '/home/lpmayos/hd/code/structural-probes/lpmayos_tests/data/en_ewt-ud-sample/kg_ctx_embs_owtrue_hid100/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logging_file = output_folder + 'logging.log'

    embeddings_extractor = EmbeddingsExtractor(logging_file, model_config)

    for file_path in input_files:
        file_path = transform_to_conllu(file_path)

        # compute embeddings

        embs = embeddings_extractor.generate_embeddings(file_path)

        # save embeddings in hdf5 format

        output_file = output_folder + file_path.split('/')[-1].replace('.conllu', '.kg-layers.hdf5')
        embeddings_extractor.save_to_hdf5(embs, output_file, skip_root=True)

        # check that embeddings were correctly saved

        embeddings_extractor.check_hdf5_file(output_file)

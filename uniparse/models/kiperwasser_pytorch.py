from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from uniparse.types import Parser


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        return out

    def get_hidden_states(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        """
        Input: Pytorch’s LSTM expects all of its inputs to be 3D tensors. The first axis is the sequence itself, the second 
        indexes instances in the mini-batch, and the third indexes elements of the input. 
                
        Source: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
        Output: 
            - the first value returned by LSTM is all of the hidden states throughout the sequence. 
            - the second is just the most recent hidden state (compare the last slice of "out" with "hidden" below, 
              they are the same).
        The reason for this is that: "out" will give you access to all hidden states in the sequence; "hidden" will 
        allow you to continue the sequence and backpropagate, by passing it as an argument  to the lstm at a later time.
        """
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        """The two 3D tensors are actually concatenated on the last axis, so to merge them, we usually do something like this:
            output = output[:, :, :self.hidden_dim] + output[:, :, self.hidden_dim:]
        aux1 = out[:, :, :self.hidden_size]
        aux2 = out[:, :, self.hidden_size:]
        aux3 = aux1 + aux2  # size torch.Size([1, 8, 100])
        """

        # Decode the hidden state of the last time step
        return out, hidden


class DependencyParser(nn.Module, Parser):
    def save_to_file(self, filename: str) -> None:
        torch.save(self.state_dict(), filename)

    def load_from_file(self, filename: str) -> None:
        self.load_state_dict(torch.load(filename))

    def __init__(self, vocab, upos_dim=25, word_dim=100, hidden_dim=100):
        super().__init__()

        upos_dim = upos_dim
        word_dim = word_dim
        input_dim = word_dim + upos_dim
        hidden_dim = hidden_dim
        num_layers = 2
        bilstm_out = hidden_dim * num_layers  # TODO lpmayos I change this; original: (word_dim+upos_dim) * 2

        self.word_count = vocab.vocab_size
        self.upos_count = vocab.upos_size
        self.i2c = defaultdict(int, vocab.wordid2freq)
        self.label_count = vocab.label_count
        self._vocab = vocab

        self.wlookup = nn.Embedding(self.word_count, word_dim)
        self.tlookup = nn.Embedding(self.word_count, upos_dim)

        self.deep_bilstm = BiRNN(input_dim, hidden_dim, num_layers)  # TODO lpmayos I change this; original: BiRNN(word_dim+upos_dim, word_dim+upos_dim, 2)

        # edge encoding
        hidden_dim_scorers = 100  # TODO lpmayos: re-set because it was originally 100, and I just want to play with the biLSTM encoder
        self.edge_head = nn.Linear(bilstm_out, hidden_dim_scorers)  # in_features, out_features
        self.edge_modi = nn.Linear(bilstm_out, hidden_dim_scorers, bias=True)

        # edge scoring
        self.e_scorer = nn.Linear(hidden_dim_scorers, 1, bias=True)

        # rel encoding
        self.label_head = nn.Linear(bilstm_out, hidden_dim_scorers)
        self.label_modi = nn.Linear(bilstm_out, hidden_dim_scorers, bias=True)

        # label scoring
        self.l_scorer = nn.Linear(hidden_dim_scorers, vocab.label_count, bias=True)

    def get_embeddings_len(self):
        return self.deep_bilstm.hidden_size

    def init_weights(self):
        nn.init.xavier_uniform_(self.wlookup.weight)
        nn.init.xavier_uniform_(self.tlookup.weight)
        nn.init.xavier_uniform_(self.plookup.weight)
        nn.init.xavier_uniform_(self.edge_head.weight)
        nn.init.xavier_uniform_(self.e_scorer.weight)
        nn.init.xavier_uniform_(self.label_head.weight)
        nn.init.xavier_uniform_(self.label_modi.weight)
        nn.init.xavier_uniform_(self.l_scorer.weight)

    def get_backend_name(self):
        return "pytorch"

    @staticmethod
    def _propability_map(matrix, dictionary):
        return np.vectorize(dictionary.__getitem__)(matrix)

    def forward(self, x):
        word_ids, lemma_ids, upos_ids, target_arcs, rel_targets, chars = x

        batch_size, n = word_ids.shape

        is_train = target_arcs is not None

        # if is_train:
            # c = self._propability_map(word_ids, self.i2c)
            # drop_mask = np.greater(0.25/(c+0.25), np.random.rand(*word_ids.shape))
            # word_ids = np.where(drop_mask, self._vocab.OOV, word_ids)  # replace with UNK / OOV

        word_embs = self.wlookup(word_ids)
        upos_embs = self.tlookup(upos_ids)

        words = torch.cat([word_embs, upos_embs], dim=-1)

        word_exprs = self.deep_bilstm(words)

        word_h = self.edge_head(word_exprs)
        word_m = self.edge_modi(word_exprs)

        arc_score_list = []
        for i in range(n):
            modifier_i = word_h[:, i, None, :] + word_m  # we would like have head major
            modifier_i = F.tanh(modifier_i)
            modifier_i_scores = self.e_scorer(modifier_i)
            arc_score_list.append(modifier_i_scores)

        arc_scores = torch.stack(arc_score_list, dim=1)
        arc_scores = arc_scores.view(batch_size, n, n)

        # Loss augmented inference
        if is_train:
            target_arcs[:, 0] = 0  # this guy contains negatives.. watch out for that ....
            margin = np.ones((batch_size, n, n))
            for bi in range(batch_size):
                for m in range(n):
                    h = target_arcs[bi, m]
                    margin[bi, m, h] -= 1
            arc_scores = arc_scores + torch.Tensor(margin)

        # since we are major
        parsed_trees = self.decode(arc_scores.transpose(1, 2))

        tree_for_rels = target_arcs if is_train else parsed_trees
        tree_for_rels[:, 0] = 0
        batch_indicies = np.repeat(np.arange(batch_size), n)  # 0, 0, 0, 0, 1, 1 ... etc
        pred_tree_tensor = tree_for_rels.reshape(-1)

        rel_heads = word_exprs[batch_indicies, pred_tree_tensor, :]
        rel_heads = self.label_head(rel_heads).view((batch_size, n, -1))
        rel_modifiers = self.label_modi(word_exprs)

        rel_arcs = F.tanh(rel_modifiers + rel_heads)

        rel_scores = self.l_scorer(rel_arcs)
        predicted_rels = rel_scores.argmax(-1).data.numpy()

        return parsed_trees, predicted_rels, arc_scores, rel_scores

    def get_hidden_states(self, word_ids, upos_ids):
        """
        """

        word_embs = self.wlookup(word_ids)
        upos_embs = self.tlookup(upos_ids)

        words = torch.cat([word_embs, upos_embs], dim=-1)

        # word_exprs = self.deep_bilstm(words)
        out, (h_n, c_n) = self.deep_bilstm.get_hidden_states(words)

        #return state_pairs_list
        return out
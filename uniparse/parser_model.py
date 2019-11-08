import os
import sys
import time
import ntpath
import logging
from typing import *

from uniparse.types import Parser
from uniparse.dataprovider import ScaledBatcher, BucketBatcher

try:
    import uniparse.decoders as decoders
except Exception as e:
    logging.error("ERROR: can't import decoders. please run 'python setup.py build_ext --inplace' from the root directory")
    raise e

import uniparse.backend as backend_wrapper
import uniparse.evaluation.universal_eval as uni_eval

import numpy as np
import sklearn.utils


class ParserModel(object):
    def __init__(self, model: Parser, decoder, loss, optimizer, strategy, vocab):
        self._model_uid = time.strftime("%m%d%H%M%S")
        self._parser = model
        self._optimizer = None
        self._vocab = vocab
        self._batch_strategy = strategy

        # retrieve backend wrapper
        self.backend = backend_wrapper.init_backend(model.get_backend_name())
        model.set_backend(self.backend)

        # extract optimizer / decoder / loss from strings
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer(optimizer)
            self._optimizer = optimizer(model.parameters())
        else:
            self._optimizer = optimizer

        # extract decoder
        runtime_decoder = self._get_decoder(decoder)
        self._parser.set_decoder(runtime_decoder)

        # extract loss functions
        self.arc_loss, self.rel_loss = self._get_loss_functions(loss)

    def _get_optimizer(self, input_optimizer):
        # get setup optimizer
        backend = self.backend
        if isinstance(input_optimizer, str):
            optimizer_options = {
                "adam": backend.optimizers.adam,
                "rmsprop": backend.optimizers.rmsprop,
                "adadelta": backend.optimizers.adadelta,
                "adagrad": backend.optimizers.adagrad
            }

            if input_optimizer not in optimizer_options:
                raise ValueError("optimizer doesn't exist")

            return optimizer_options[input_optimizer]
        else:
            return input_optimizer

    @staticmethod
    def _get_decoder(input_decoder):
        if isinstance(input_decoder, str):
            decoder_options = {
                "eisner": decoders.eisner,
                "cle": decoders.cle
            }

            if input_decoder not in decoder_options:
                raise ValueError("decoder (%s) not available" % input_decoder)

            return decoder_options[input_decoder]
        else:
            return input_decoder

    def _get_loss_functions(self, input_loss: Union[str, Tuple[Any, Any]]):
        if isinstance(input_loss, str):
            loss = self.backend.loss
            loss_options = {
                # included for completeness
                "crossentropy": (loss.crossentropy, loss.crossentropy),
                "kiperwasser": (loss.hinge, loss.hinge),
                "hinge": (loss.hinge, loss.hinge)
            }
            if input_loss not in loss_options:
                raise ValueError("unknown loss function %s" % input_loss)

            return loss_options[input_loss]
        else:
            return input_loss

    def _batch_data(self, samples: List, strategy: str, scale: int, shuffle: bool):
        if strategy == "bucket":
            dataprovider = BucketBatcher(samples, padding_token=self._vocab.PAD)
            _idx, _sentences = dataprovider.get_data(scale, shuffle)
        elif strategy == "scaled_batch":
            dataprovider = ScaledBatcher(samples, cluster_count=40, padding_token=self._vocab.PAD)
            _idx, _sentences = dataprovider.get_data(scale, shuffle)
        else:
            raise ValueError("no such data strategy")

        return _idx, _sentences

    def extract_embeddings(self, samples: List, format='concat'):
        """
        based on original function 'run', but instead of calling _parser() to end up executing dynet 'transduce'
        function, we call 'extract_internal_states' to end up executing dynet 'add_inputs' function:
            add_inputs(es)
                returns the list of state pairs (stateF, stateB) obtained by adding inputs to both forward (stateF) and
                backward (stateB) RNNs. Does not preserve the internal state after adding the inputs
        """
        backend = self.backend

        total_words = sum([len(a[0]) for a in samples])
        embeddings_per_word = 4
        embeddings_len = self._parser.get_embeddings_len()
        embeddings = np.zeros((total_words, embeddings_per_word, embeddings_len))

        i = 0
        for sample in samples:
            backend.renew_cg()

            words, lemmas, tags, heads, rels, chars = sample

            words = backend.input_tensor(np.array([words]), dtype="int")
            tags = backend.input_tensor(np.array([tags]), dtype="int")

            states = self._parser.extract_internal_states(words, tags)

            for state in states:  # we receive one state per each word in the sample
                state_layer1 = state[0]
                hidden_state_layer1 = state_layer1.s()
                hidden_state_layer1_f = np.array(hidden_state_layer1[0].value())
                hidden_state_layer1_b = np.array(hidden_state_layer1[1].value())

                state_layer2 = state[1]
                hidden_state_layer2 = state_layer2.s()
                hidden_state_layer2_f = np.array(hidden_state_layer2[0].value())
                hidden_state_layer2_b = np.array(hidden_state_layer2[1].value())

                embeddings[i][0] = hidden_state_layer1_f
                embeddings[i][1] = hidden_state_layer1_b
                embeddings[i][2] = hidden_state_layer2_f
                embeddings[i][3] = hidden_state_layer2_b

                i += 1

        # at this point embeddings is a numpy array with shape n_words x 4 x 125

        if format == 'average':
            raise NotImplementedError
        elif format == 'max':
            raise NotImplementedError
        else:  # default: concat
            embeddings_dimension = embeddings.shape[1] * embeddings.shape[2]
            embeddings = embeddings.reshape((embeddings.shape[0], embeddings_dimension))

        embeddings = embeddings.astype(np.float32)
        return embeddings

    def extract_embeddings_from_word_tags_tuple(self, sentence_words):
        sentence_tags = []  # TODO we assume that we work always with the only_words=True model; rethink
        batch_size = 32  # TODO hardcoded for testing purposes; see if we can determine it any other way
        input_data = self._vocab.word_tags_tuple_to_conll(sentence_words, sentence_tags)
        contextual_embeddings = self.extract_embeddings(input_data, batch_size)
        del input_data
        return contextual_embeddings

    def run(self, samples: List, batch_size: int):
        indices, batches = self._batch_data(samples, strategy=self._batch_strategy, scale=batch_size, shuffle=False)
        backend = self.backend

        predictions = []
        for indicies, (x, y) in zip(indices, batches):
            backend.renew_cg()

            words, lemmas, tags, chars = x

            words = backend.input_tensor(words, dtype="int")
            tags = backend.input_tensor(tags, dtype="int")
            lemmas = backend.input_tensor(lemmas, dtype="int")

            arc_preds, rel_preds, _, _ = self._parser((words, lemmas, tags, None, None, chars))

            outs = [(ind, arc[1:], rel[1:]) for ind, arc, rel in zip(indicies, arc_preds, rel_preds)]

            predictions.extend(outs)

        predictions.sort(key=lambda tup: tup[0])

        return predictions

    def train(self, train: List, dev_file: str, dev: List, epochs: int, batch_size: int, callbacks: List = None, patience:int = -1):
        callbacks = callbacks if callbacks else []  # This is done to avoid using the same list.
        
        if patience > -1:
            logging.debug(f"...Training with patience {patience} for less than {epochs} epochs")
        else: 
            logging.debug(f"...Training without patience for exactly {epochs} epochs")

        running_patience = patience

        training_data = self._batch_data(train, strategy=self._batch_strategy, scale=batch_size, shuffle=True)
        '''
        i.e. in dev mode, train is a list of len 100.
        each element in train is a tuple of 6 elements: ([words], [lemmas], ...) --> ([1, 452, 12188, 3107, 19765, 5], [1, 2, 2, 2, 2, 2], [1, 3, 3, 11, 3, 4], [-1, 2, 3, 0, 3, 3], [1, 42, 19, 1, 12, 3], [[1], [18, 57, 39], [40, 52, 52, 24], [81, 15, 52, 16, 57], [11, 15, 79, 52, 27, 46, 79], [39]])
        
        training_data is a tuple of 2 elements: _idx, _sentences --> both elements are lists
        _idx = [[0, 80], [1], [92, 2, 88], [58, 3, 74], [4, 90, 35, 50, 98], [5, 20, 75], [21, 6, 76], [83, 7], [94, 8, 12], [9, 72], [10, 62, 36, 82, 81, 48], [11], [22, 13], [14, 52], [15], [16], [24, 78, 17, 57, 95, 38, 33], [68, 18], [19, 29], [71, 23, 66], [25, 39], [51, 26], [27, 91, 93, 37], [28, 31, 42], [30], [32, 49, 43, 59], [34, 65, 45, 56, 47, 60], [40, 87], [96, 41], [44], [46, 70], [53, 89], [54, 73], [55, 61], [63], [64], [67], [69], [77], [79], [84], [85], [86], [97], [99]]
        _sentences = [Batch1, Batch2, Batch3... ]
        
        
        '''

        backend = self.backend
        _, samples = training_data
        global_step = 0
        max_dev_uas=0.0
        for epoch in range(1, epochs+1):
            start = time.time()

            samples = sklearn.utils.shuffle(samples)

            logging.info(f"Epoch {epoch}")
            logging.info("=====================")

            for step, (x, y) in enumerate(samples):
                batch_size, global_step = self._train_step(backend, batch_size, callbacks, global_step, x, y)

            do_break = self._evaluate_epoch(epoch, dev, dev_file, callbacks, batch_size, patience, max_dev_uas, running_patience, global_step, start)
            if do_break:
                break

        logging.debug(f"Finished at epoch {epoch}")

    def train_big_datasets(self, train_file: str, dev_file: str, dev: List, epochs: int, batch_size: int, callbacks: List = None, patience: int = -1, subset_size: int = 100000):
        callbacks = callbacks if callbacks else []  # This is done to avoid using the same list.

        if patience > -1:
            logging.debug(f"...Training with patience {patience} for less than {epochs} epochs")
        else:
            logging.debug(f"...Training without patience for exactly {epochs} epochs")

        running_patience = patience

        backend = self.backend
        global_step = 0
        max_dev_uas = 0.0
        for epoch in range(1, epochs + 1):

            start = time.time()
            logging.info("")
            logging.info(f"Epoch {epoch}")
            logging.info("=====================")

            with open(train_file, encoding="UTF-8") as f:

                # ---------------------------------------------------
                # I move here functionality from vocabulary.py for the sake of efficiency in the large file reading (lpmayos)

                tokenize = True

                word_root = self._vocab.ROOT
                lemma_root = self._vocab.ROOT
                tag_root = self._vocab.ROOT
                rel_root = self._vocab.ROOT
                char_root = [self._vocab.ROOT]
                root_head = -1

                words, lemmas, tags, heads, rels, chars = [word_root], [lemma_root], [tag_root], [root_head], [rel_root], [char_root]

                read_sentences = 0
                total_read_sentences = 0
                training_data = []
                drop_sentence = False
                for line in f.readlines():

                    try:
                        blank_line, comment_line, word, lemma, tag, head, rel, characters = self._vocab._parse_line(line, tokenize=tokenize)
                    except:
                        drop_sentence = True

                    if comment_line:
                        pass

                    elif not blank_line:
                        words.append(word)
                        lemmas.append(lemma)
                        tags.append(tag)
                        heads.append(head)
                        rels.append(rel)
                        chars.append(characters)

                    else:
                        sent = (words, lemmas, tags, heads, rels, chars)
                        if not drop_sentence:
                            training_data.append(sent)
                            read_sentences += 1
                            total_read_sentences += 1
                        drop_sentence = False
                        words, lemmas, tags, heads, rels, chars = [word_root], [lemma_root], [tag_root], [root_head], [rel_root], [char_root]

                    if read_sentences > 0 and read_sentences % subset_size == 0:  # we have read 10000 sentences, lets use them to train

                        logging.info('train_big_datasets; epoch %s; total sentences used to train: %s; read_sentences %s' % (epoch, total_read_sentences, read_sentences))

                        _, samples = self._batch_data(training_data, strategy=self._batch_strategy, scale=batch_size, shuffle=True)
                        samples = sklearn.utils.shuffle(samples)

                        for step, (x, y) in enumerate(samples):

                            batch_size, global_step = self._train_step(backend, batch_size, callbacks, global_step, x, y)

                        read_sentences = 0
                        training_data = []

                if len(training_data) > 0:  # train with the last sentences
                    logging.info('train_big_datasets; epoch %s; total sentences used to train: %s; read_sentences %s' % (epoch, total_read_sentences, read_sentences))

                    _, samples = self._batch_data(training_data, strategy=self._batch_strategy, scale=batch_size, shuffle=True)
                    samples = sklearn.utils.shuffle(samples)

                    for step, (x, y) in enumerate(samples):
                        batch_size, global_step = self._train_step(backend, batch_size, callbacks, global_step, x, y)

                # we have trained with all the sentences of the training set; evaluate epoch and finish, if needed
                do_break = self._evaluate_epoch(epoch, dev, dev_file, callbacks, batch_size, patience, max_dev_uas, running_patience, global_step, start)
                if do_break:
                    break

            f.close()

        logging.debug(f"Finished at epoch {epoch}")

    def _train_step(self, backend, batch_size, callbacks, global_step, x, y):

        # renew graph
        backend.renew_cg()

        words, lemmas, tags, chars = x
        gold_arcs, gold_rels = y

        batch_size, n = words.shape

        # PAD = 0; ROOT = 1; OOV = 2; UNK = 2
        # Tokens > 1 are valid tokens we want to compute loss for use for accuracy metrics
        mask = np.greater(words, self._vocab.ROOT)
        num_tokens = int(np.sum(mask))

        """ this is necessary for satisfy compatibility with pytorch """
        words = backend.input_tensor(words, dtype="int")
        postags = backend.input_tensor(tags, dtype="int")
        lemmas = backend.input_tensor(lemmas, dtype="int")

        arc_preds, rel_preds, arc_scores, rel_scores = self._parser((words, lemmas, postags, gold_arcs, gold_rels, chars))

        arc_loss = self.arc_loss(arc_scores, arc_preds, gold_arcs, mask)
        rel_loss = self.rel_loss(rel_scores, None, gold_rels, mask)

        loss = arc_loss + rel_loss
        loss_value = backend.get_scalar(loss)
        loss.backward()

        backend.step(self._optimizer)

        arc_correct = np.equal(arc_preds, gold_arcs).astype(np.float32) * mask
        arc_accuracy = np.sum(arc_correct) / num_tokens

        rel_correct = np.equal(rel_preds, gold_rels).astype(np.float32) * mask
        rel_accuracy = np.sum(rel_correct) / num_tokens

        training_info = {
            "arc_accuracy": arc_accuracy,
            "rel_accuracy": rel_accuracy,
            "arc_loss": backend.get_scalar(arc_loss),
            "rel_loss": backend.get_scalar(rel_loss),
            "global_step": global_step
        }

        for callback in callbacks:
            callback.on_batch_end(training_info)

        sys.stdout.write(
            "\r\rStep #%d: Acc: arc %.2f, rel %.2f, loss %.3f"
            % (global_step, float(arc_accuracy), float(rel_accuracy), loss_value)
        )
        sys.stdout.flush()

        global_step += 1

        return batch_size, global_step

    def _evaluate_epoch(self, epoch, dev, dev_file, callbacks, batch_size, patience, max_dev_uas, running_patience, global_step, start):
        logging.debug("Completed epoch %s in %s" % (epoch, time.time() - start))

        do_break = False

        metrics = self.parse_and_evaluate(dev_file, dev, batch_size, None)
        no_punct_dev_uas = metrics["nopunct_uas"]
        no_punct_dev_las = metrics["nopunct_las"]
        punct_dev_uas = metrics["uas"]
        punct_dev_las = metrics["las"]
        logging.debug(f"UAS (wo. punct) {no_punct_dev_uas:.{5}}\t LAS (wo. punct) {no_punct_dev_las:.{5}}")
        logging.debug(f"UAS (w. punct) {punct_dev_uas:.{5}}\t LAS (w. punct) {punct_dev_las:.{5}}")

        if patience > -1:
            if max_dev_uas > no_punct_dev_uas:
                max_dev_uas = no_punct_dev_uas
                running_patience -= 1
                logging.debug(f"Patience decremented to {running_patience}")
            else:
                running_patience = patience
                logging.debug(f"Patience incremented to {running_patience}")

            if running_patience == 0:
                do_break = True
                return do_break

        batch_end_info = {
            "dev_uas": no_punct_dev_uas,
            "dev_las": no_punct_dev_las,
            "global_step": global_step,
            "model": self._parser
        }

        for callback in callbacks:
            callback.on_epoch_end(epoch, batch_end_info)

        return do_break



    def parse(self, test_file: str, test_data: List, batch_size: int, output_file: str):

        temporal = False
        if output_file is None:
            stripped_filename = ntpath.basename(test_file)
            output_file = f"{self._model_uid}_on_{stripped_filename}"
            temporal = True

        # run parser on data
        predictions = self.run(test_data, batch_size)

        # write to file
        uni_eval.write_predictions_to_file(predictions, reference_file=test_file, output_file=output_file, vocab=self._vocab)
        logging.debug('output file saved to %s' % (output_file))

        return output_file, temporal

    def evaluate(self, output_file, test_file):

        metrics = uni_eval.evaluate_files(output_file, test_file)
        return metrics

    def parse_and_evaluate(self, test_file: str, test_data: List, batch_size: int, output_file: str):

        output_file, temporal = self.parse(test_file, test_data, batch_size, output_file)
        metrics = uni_eval.evaluate_files(output_file, test_file)

        if temporal:
            os.remove(output_file)

        return metrics

    def save_to_file(self, filename: str) -> None:
        self._parser.save_to_file(filename)

    def load_from_file(self, filename: str) -> None:
        self._parser.load_from_file(filename)

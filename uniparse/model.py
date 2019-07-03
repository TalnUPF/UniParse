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


class Model(object):
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
                "kiperwasser": (loss.hinge, loss.kipperwasser_hinge),
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

        running_patience=patience

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

            logging.debug("Completed epoch %s in %s" % (epoch, time.time()-start))
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
                    running_patience-=1
                    logging.debug(f"Patience decremented to {running_patience}")
                else:
                    running_patience=patience
                    logging.debug(f"Patience incremented to {running_patience}")
                
                if running_patience==0:
                    break    
            

            batch_end_info = {
                "dev_uas": no_punct_dev_uas,
                "dev_las": no_punct_dev_las,
                "global_step": global_step,
                "model": self._parser
            }

            for callback in callbacks:
                callback.on_epoch_end(epoch, batch_end_info)

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

            init_sent = 0
            end_sent = init_sent + subset_size - 1
            training_data = self._vocab.tokenize_conll(train_file, init_sent, end_sent)  # we just tokenize the sentences we need for training

            while len(training_data) > 0:

                _, samples = self._batch_data(training_data, strategy=self._batch_strategy, scale=batch_size, shuffle=True)
                samples = sklearn.utils.shuffle(samples)

                for step, (x, y) in enumerate(samples):

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

                init_sent = end_sent + 1
                end_sent = init_sent + subset_size -1
                training_data = self._vocab.tokenize_conll(train_file, init_sent, end_sent)  # we just tokenize the sentences we need for training


            logging.debug("Completed epoch %s in %s" % (epoch, time.time() - start))

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
                    break

            batch_end_info = {
                "dev_uas": no_punct_dev_uas,
                "dev_las": no_punct_dev_las,
                "global_step": global_step,
                "model": self._parser
            }

            for callback in callbacks:
                callback.on_epoch_end(epoch, batch_end_info)


        logging.debug(f"Finished at epoch {epoch}")

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

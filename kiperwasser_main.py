import argparse
import logging
import os

from uniparse import Vocabulary, ParserModel
from uniparse.callbacks import TensorboardLoggerCallback, ModelSaveCallback
from uniparse.models.kiperwasser import DependencyParser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_or_create_vocab_and_embs(arguments):

    vocab = Vocabulary(arguments.only_words)

    # load or create vocabulary
    try:
        vocab.load(arguments.vocab_file)
    except:
        if arguments.embs == None:
            vocab = vocab.fit(arguments.train)
        else:
            vocab = vocab.fit(arguments.train, arguments.embs)

        # save vocab for reproducability later
        logging.info("> saving vocab to %s" % (arguments.vocab_file))
        vocab.save(arguments.vocab_file)

    if arguments.embs == None:
        embs = None
    else:
        embs = vocab.load_embedding()
        logging.info('shape %s' % (embs.shape))

    return vocab, embs


def do_training(arguments, vocab, embs):
    logging.debug("Init training")
    n_epochs = arguments.epochs
    batch_size = arguments.batch_size

    # prep data
    logging.info(">> Loading in data")

    logging.info("tokenizing train data ...")
    training_data = vocab.tokenize_conll(arguments.train)
    logging.info("... tokenized train data")

    if arguments.dev_mode:
        training_data=training_data[:100]

    logging.info("tokenizing dev data ...")
    dev_data = vocab.tokenize_conll(arguments.dev)
    logging.info("... tokenized dev data")

    # instantiate model
    logging.info("creating model ...")
    model = DependencyParser(vocab, embs, arguments.no_update_pretrained_emb)
    logging.info("... model created")

    callbacks = []
    tensorboard_logger = None
    if arguments.tb_dest:
        tensorboard_logger = TensorboardLoggerCallback(arguments.tb_dest)
        callbacks.append(tensorboard_logger)


    logging.info("creating ModelSaveCallback ...")
    save_callback = ModelSaveCallback(arguments.model_file)
    callbacks.append(save_callback)
    logging.info("... ModelSaveCallback created")

    # prep params
    logging.info("creating Model ...")
    parser = ParserModel(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    logging.info("... Model created")

    logging.info("training Model ...")
    parser.train(training_data, arguments.dev, dev_data, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks, patience=arguments.patience)
    logging.info("...Model trained")

    logging.info("Model maxed on dev at epoch %s " % (save_callback.best_epoch))

    return parser


def do_training_big_datasets(arguments, vocab, embs, subset_size):
    logging.debug("Init training with big dataset (there is no dev mode)")
    n_epochs = arguments.epochs
    batch_size = arguments.batch_size

    logging.info("tokenizing dev data ...")
    dev_data = vocab.tokenize_conll(arguments.dev)
    logging.info("... tokenized dev data")

    # instantiate model
    logging.info("creating model ...")
    model = DependencyParser(vocab, embs, arguments.no_update_pretrained_emb)
    logging.info("... model created")

    callbacks = []
    if arguments.tb_dest:
        tensorboard_logger = TensorboardLoggerCallback(arguments.tb_dest)
        callbacks.append(tensorboard_logger)

    logging.info("creating ModelSaveCallback ...")
    save_callback = ModelSaveCallback(arguments.model_file)
    callbacks.append(save_callback)
    logging.info("... ModelSaveCallback created")

    # prep params
    logging.info("creating Model ...")
    parser = ParserModel(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    logging.info("... Model created")

    logging.info("training Model ...")
    parser.train_big_datasets(arguments.train, arguments.dev, dev_data, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks, patience=arguments.patience, subset_size=subset_size)
    logging.info("...Model trained")

    logging.info("Model maxed on dev at epoch %s " % (save_callback.best_epoch))

    return parser


def create_new_conllu(original_file):

    def _line_to_conllu(line, id):
        """
        ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).
        FORM: Word form or punctuation symbol.
        LEMMA: Lemma or stem of word form.
        UPOS: Universal part-of-speech tag.
        XPOS: Language-specific part-of-speech tag; underscore if not available.
        FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
        HEAD: Head of the current word, which is either a value of ID or zero (0).
        DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
        DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
        MISC: Any other annotation.

        """
        info = line.strip().split()

        if len(info) == 2:  # dep parse line; i.e. the 7-det
            form = info[0]
            head = info[1].split('-')[0]
            rel = info[1].split('-')[1]

        elif len(info) == 3:  # chunking line; i.e. said VBD B-VP
            form = info[0]
            head = -1
            rel = '_'

        return '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (id, form, '_', '_', '_', '_', head, rel, '_', '_')

    new_file_name = original_file.replace('.txt', '.txt.conllu')

    with open(original_file, encoding="UTF-8") as f, open(new_file_name, 'w+', encoding="UTF-8") as f_new:
        id = 1
        for line in f:
            comment_line = line.startswith('#')
            blank_line = line == "\n"

            if comment_line or blank_line:
                new_line = line
                id = 1
            else:
                new_line = _line_to_conllu(line, id)
                id += 1

            f_new.write(new_line)

    logging.info('Created new file %s from the original file %s' % (new_file_name, original_file))

    return new_file_name


def transform_to_conllu(filename):
    """
    Transform cvt_txt text input files into conllu that we can use.
    """
    if filename.endswith('.txt'):
        filename = create_new_conllu(filename)
    return filename


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--only_words", dest="only_words", type=str2bool, default=False, help="Should we use only words to train? Lemmas and POS will be ignored", required=True)

    parser.add_argument("--do_training", dest="do_training", type=str2bool, default=False, help="Should we train the model?", required=True)
    parser.add_argument("--train_file", dest="train", help="Annotated CONLL train file", metavar="FILE", required=False)
    parser.add_argument("--dev_file", dest="dev", help="Annotated CONLL dev file", metavar="FILE", required=False)
    parser.add_argument("--test_file", dest="test", help="Annotated CONLL dev test", metavar="FILE", required=True)

    parser.add_argument("--results_folder", dest="results_folder", help="Folder to store log, model, vocabulary and output", metavar="FILE", required=True)
    parser.add_argument("--logging_file", dest="logging_file", help="File to store the logs", metavar="FILE", required=True)
    parser.add_argument("--output_file", dest="output_file", help="CONLL output file", metavar="FILE", required=True)
    parser.add_argument("--vocab_file", dest="vocab_file", required=True)
    parser.add_argument("--model_file", dest="model_file", required=True)

    parser.add_argument("--epochs", dest="epochs", type=int, default=30)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--tb_dest", dest="tb_dest")
    parser.add_argument("--embs", dest="embs", help="pre-trained embeddings file name", required=False)
    parser.add_argument("--no_update_pretrained_emb", dest="no_update_pretrained_emb", type=str2bool, default=False, help="don't update the pretrained embeddings during training")
    parser.add_argument("--patience", dest='patience', type=int, default=-1)
    parser.add_argument("--dev_mode", dest='dev_mode', type=str2bool, default=False, help='small subset of training examples, for code testing')

    parser.add_argument("--big_dataset", dest='big_dataset', type=str2bool, default=False, help='Are you training with a huge dataset? (i.e. 1B benchmark)')

    arguments, unknown = parser.parse_known_args()


    # create results folder if needed

    if not os.path.exists(arguments.results_folder):
        os.makedirs(arguments.results_folder)


    # configure logging

    logging.basicConfig(filename=arguments.logging_file,
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:\t%(message)s")


    logging.info("\n\n\n===================================================================================================")
    logging.info("kiperwasser_main")
    logging.info("===================================================================================================\n")

    logging.info("\nArguments:")
    logging.info(arguments)
    logging.info("\n")

    # load or create vocabulary and embeddings

    vocab, embs = load_or_create_vocab_and_embs(arguments)

    # transform input files into conllu if needed

    arguments.train = transform_to_conllu(arguments.train)
    arguments.dev = transform_to_conllu(arguments.dev)
    arguments.test = transform_to_conllu(arguments.test)


    # load or train parser

    if arguments.do_training:
        if not arguments.big_dataset:
            logging.info('Training with normal dataset')
            parser = do_training(arguments, vocab, embs)
        else:
            subset_size = 10000
            logging.info('Training with big dataset; subset_size = %i' % subset_size)
            parser = do_training_big_datasets(arguments, vocab, embs, subset_size)

    else:
        logging.info('No training; loading model')
        model = DependencyParser(vocab, embs, arguments.no_update_pretrained_emb)
        parser = ParserModel(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)

    parser.load_from_file(arguments.model_file)

    # parse test file

    test_data = vocab.tokenize_conll(arguments.test)
    output_file, temporal = parser.parse(arguments.test, test_data, arguments.batch_size, arguments.output_file)

    # evaluate output

    metrics = parser.evaluate(output_file, arguments.test)
    test_UAS = metrics["nopunct_uas"]
    test_LAS = metrics["nopunct_las"]

    logging.info(metrics)

    if arguments.tb_dest and tensorboard_logger:
        tensorboard_logger.raw_write("test_UAS", test_UAS)
        tensorboard_logger.raw_write("test_LAS", test_LAS)

    logging.info("\n--------------------------------------------------------")
    logging.info("Test score: %s %s" % (test_UAS, test_LAS))
    logging.info("--------------------------------------------------------\n")


if __name__ == '__main__':
    main()

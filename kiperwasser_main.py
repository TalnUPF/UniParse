import argparse
import logging

from uniparse import Vocabulary, Model
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

    vocab_file = "%s/%s" % (arguments.results_folder, arguments.vocab_file)

    if arguments.do_training:

        vocab = Vocabulary()
        if arguments.embs == None:
            vocab = vocab.fit(arguments.train)
            embs = None
        else:
            vocab = vocab.fit(arguments.train, arguments.embs)
            embs = vocab.load_embedding()
            logging.info('shape %s' % (embs.shape))

        # save vocab for reproducability later
        logging.info("> saving vocab to %s" % (vocab_file))
        vocab.save(vocab_file)

    else:

        vocab = Vocabulary()
        vocab.load(vocab_file)

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
    training_data = vocab.tokenize_conll(arguments.train)
    if arguments.dev_mode:
        training_data=training_data[:100]
    dev_data = vocab.tokenize_conll(arguments.dev)

    # instantiate model
    model = DependencyParser(vocab, embs, arguments.no_update_pretrained_emb)

    callbacks = []
    tensorboard_logger = None
    if arguments.tb_dest:
        tensorboard_logger = TensorboardLoggerCallback(arguments.tb_dest)
        callbacks.append(tensorboard_logger)


    save_callback = ModelSaveCallback("%s/%s" % (arguments.results_folder, arguments.model_file))
    callbacks.append(save_callback)

    # prep params
    parser = Model(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    parser.train(training_data, arguments.dev, dev_data, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks, patience=arguments.patience)

    logging.info("Model maxed on dev at epoch %s " % (save_callback.best_epoch))

    return parser


def main():
    """
    train sample call:
      $ python kiperwasser_main.py --results_folder /home/lpmayos/code/UniParse/saved_models/model_en_ud_test --logging_file logging.log --do_training True --train_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-train.conllu --dev_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-dev.conllu --test_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-test.conllu --output_file prova.output --model_file model.model --vocab_file vocab.pkl --dev_mode True

    test sample call:
      $ python kiperwasser_main.py --results_folder /home/lpmayos/code/UniParse/saved_models/model_en_ud --logging_file logging.log --do_training False --test_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-test.conllu --output_file prova2.output --model_file model.model --vocab_file vocab.pkl --dev_mode True
    """

    parser = argparse.ArgumentParser()

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

    arguments, unknown = parser.parse_known_args()

    # configure logging

    logging.basicConfig(filename="%s/%s" % (arguments.results_folder, arguments.logging_file),
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:\t%(message)s")

    logging.info("")
    logging.info("")
    logging.info("")
    logging.info("===================================================================================================")
    logging.info("kiperwasser_main")
    logging.info("===================================================================================================")
    logging.info("")

    # load or create vocabulary and embeddings

    vocab, embs = load_or_create_vocab_and_embs(arguments)

    # create parser and train it if needed

    if arguments.do_training:
        parser = do_training(arguments, vocab, embs)

    else:
        model = DependencyParser(vocab, embs, arguments.no_update_pretrained_emb)
        parser = Model(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)

    parser.load_from_file("%s/%s" % (arguments.results_folder, arguments.model_file))

    # test

    test_data = vocab.tokenize_conll(arguments.test)

    metrics = parser.evaluate(arguments.test, test_data, arguments.batch_size, "%s/%s" % (arguments.results_folder, arguments.output_file))
    test_UAS = metrics["nopunct_uas"]
    test_LAS = metrics["nopunct_las"]

    logging.info(metrics)

    if arguments.tb_dest and tensorboard_logger:
        tensorboard_logger.raw_write("test_UAS", test_UAS)
        tensorboard_logger.raw_write("test_LAS", test_LAS)

    logging.info("")
    logging.info("--------------------------------------------------------")
    logging.info("Test score: %s %s" % (test_UAS, test_LAS))
    logging.info("--------------------------------------------------------")
    logging.info("")


if __name__ == '__main__':
    main()

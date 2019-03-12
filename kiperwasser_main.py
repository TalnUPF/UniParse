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
        logging.info("> saving vocab to %s" % (arguments.vocab_file))
        vocab.save(arguments.vocab_file)

    else:

        vocab = Vocabulary()
        vocab.load(arguments.vocab_file)

        if arguments.embs == None:
            embs = None
        else:
            embs = vocab.load_embedding()
            logging.info('shape %s' % (embs.shape))

    return vocab, embs


def do_training(arguments, vocab, embs):
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


    save_callback = ModelSaveCallback(arguments.model_file)
    callbacks.append(save_callback)

    # prep params
    parser = Model(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)
    parser.train(training_data, arguments.dev, dev_data, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks, patience=arguments.patience)

    logging.info("\n>>> Model maxed on dev at epoch %s " % (save_callback.best_epoch))

    return parser


def main():
    parser = argparse.ArgumentParser()

    # train sample call:
    #   $ python kiperwasser_main.py --do_training True --save_output False --train_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-train.conllu --dev_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-dev.conllu --test_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-test.conllu --model_file /home/lpmayos/code/UniParse/saved_models/model_en_ud/model.model --vocab_file /home/lpmayos/code/UniParse/saved_models/model_en_ud/vocab.pkl --dev_mode True
    # test sample call:
    #   $ python kiperwasser_main.py --do_training False --vocab_file /home/lpmayos/code/UniParse/saved_models/model_en_ud/vocab.pkl --save_output True --output_file /home/lpmayos/code/UniParse/saved_models/model_en_ud/prova.output --test_file /home/lpmayos/code/datasets/ud2.1/ud-treebanks-v2.1/UD_English/en-ud-test.conllu --model_file /home/lpmayos/code/UniParse/saved_models/model_en_ud/model.model

    parser.add_argument("--do_training", dest="do_training", type=str2bool, default=False, help="Should we train the model?", required=True)
    parser.add_argument("--train_file", dest="train", help="Annotated CONLL train file", metavar="FILE", required=False)
    parser.add_argument("--dev_file", dest="dev", help="Annotated CONLL dev file", metavar="FILE", required=False)
    parser.add_argument("--test_file", dest="test", help="Annotated CONLL dev test", metavar="FILE", required=True)
    parser.add_argument("--save_output", dest="save_output", type=str2bool, default=False, help="Should we saved the result of the parsing?", required=True)
    parser.add_argument("--output_file", dest="output_file", help="CONLL output file", metavar="FILE", required=False)
    parser.add_argument("--epochs", dest="epochs", type=int, default=30)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--tb_dest", dest="tb_dest")
    parser.add_argument("--vocab_file", dest="vocab_file", required=True)
    parser.add_argument("--model_file", dest="model_file", required=True)
    parser.add_argument("--embs", dest="embs", help="pre-trained embeddings file name", required=False)
    parser.add_argument("--no_update_pretrained_emb", dest="no_update_pretrained_emb", type=str2bool, default=False, help="don't update the pretrained embeddings during training")
    parser.add_argument("--patience", dest='patience', type=int, default=-1)
    parser.add_argument("--dev_mode", dest='dev_mode', type=str2bool, default=False, help='small subset of training examples, for code testing')

    arguments, unknown = parser.parse_known_args()

    vocab, embs = load_or_create_vocab_and_embs(arguments)

    if arguments.do_training:
        parser = do_training(arguments, vocab, embs)

    else:
        model = DependencyParser(vocab, embs, arguments.no_update_pretrained_emb)
        parser = Model(model, decoder="eisner", loss="kiperwasser", optimizer="adam", strategy="bucket", vocab=vocab)

    parser.load_from_file(arguments.model_file)

    test_data = vocab.tokenize_conll(arguments.test)

    metrics = parser.evaluate(arguments.test, test_data, arguments.batch_size, arguments.save_output, arguments.output_file)
    test_UAS = metrics["nopunct_uas"]
    test_LAS = metrics["nopunct_las"]

    logging.info(metrics)

    if arguments.tb_dest and tensorboard_logger:
        tensorboard_logger.raw_write("test_UAS", test_UAS)
        tensorboard_logger.raw_write("test_LAS", test_LAS)

    logging.info(">>> Test score: %s %s" % (test_UAS, test_LAS))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()

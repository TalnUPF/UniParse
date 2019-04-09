import logging
import sentencepiece as spm


if __name__ == "__main__":
    """ 
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s:\t%(message)s")


    # 1. train bpe model (bpe.sh) with [raw 1b text training set + raw penn 
    #    training set] as input, 80k vocab --> bpe.model; bpe.vocab

    # 2. encode penn (all files) using bpe.model

    # 3. reconstruct penn dependency trees: iterate simultaneously over penn 
    #    enocoded files and conll files, splitting lines when necessary (if 
    #    a word is divided in two, first part is head and rest is dependant).
    #    --> penn_train_pieced.conll, penn_dev_pieced.conll and 
    #        penn_test_pieced.conll

    # 4. train kiperwasser/StanfordCoreNLP parser with penn_train_pieced.conll, 
    #    penn_dev_pieced.conll and penn_test_pieced.conll --> model1_pieced

    # 5. use kiperwasser/StanfordCoreNLP + model1_pieced to parse 
    #    1B input_pieced.txt[1b] --> 1b_train_pieced.conll, 1b_dev_pieced.conll, 
    #    1b_test_pieced.conll

    # 6. train kiperwasser parser with 1b_train_pieced.conll, 
    #    1b_dev_pieced.conll, 1b_test_pieced --> model3
import logging
from io import open
from conllu import parse_incr, TokenList
import sentencepiece as spm
from collections import OrderedDict


def conllu_to_text(input_file_path, output_file_path):
    """ reads input_file_path conllu file and saves raw text sentences to output_file_path
    """
    conllu_file = open(input_file_path, "r", encoding="utf-8")
    text_file = open(output_file_path, "w", encoding="utf-8")
    for sentence in parse_incr(conllu_file):
        text_file.write(' '.join([a['form'] for a in sentence]))
        text_file.write('\n')


def reconstruct_dependency_trees(conll_file_path, model_prefix, output_file_path):
    """ adds extra conll lines for tokens that have been splitted by sentencepiece
    and changes original ids (and head pointers) accordingly
    """

    def _get_correspondances(pieces, sentence):
        """ returns a list of tuples with the correspondance between the pieces 
        and the tokens in sentence; i.e. [([0], [0]), ([1], [1]), ([2], [2]), ([3, 4], [3]), ([5], [4])]
        """

        correspondances = []
        i_pieces = 0
        i_sentence = 0
        while i_pieces < len(pieces) and i_sentence < len(sentence):
            piece = pieces[i_pieces].replace('▁', '')
            token = sentence[i_sentence]['form'].replace('\xa0', ' ')  # \xa0 is actually non-breaking space in Latin1 (ISO 8859-1)

            if piece == token or len(piece) >= len(token):  # second condiction added to cover ™/TM cases
                correspondance = ([i_pieces], [i_sentence])

            elif len(token) > len(piece): 
                # add pieces until they are the same
                correspondance = ([i_pieces], [i_sentence])
                while piece != token and i_pieces < len(pieces) - 1:
                    i_pieces += 1
                    new_piece = pieces[i_pieces].replace('▁', ' ')
                    piece += new_piece
                    correspondance[0].append(i_pieces)

            i_pieces += 1
            i_sentence += 1
            correspondances.append(correspondance)

        return correspondances

    def _increment_ids(sentence, tokens_to_add, increment, origin):
        """ increments in 'increment' the id of all tokens with id > origin;
        increments in 'increment' the head of all tokens with head > origin;
        """
        for token in sentence:
            if token['id'] > origin:
                token['id'] += increment
            if token['head'] > origin:
                token['head'] += increment

        for token in tokens_to_add:
            if token['id'] > origin:
                token['id'] += increment
            if token['head'] > origin:
                token['head'] += increment

    conll_file = open(conll_file_path, "r", encoding="utf-8")
    output_file = open(output_file_path, "w", encoding="utf-8")

    conll_sentences = parse_incr(conll_file)

    sp = spm.SentencePieceProcessor()                                                                                                                                                                            
    sp.Load('%s.model' % model_prefix)

    for sentence in conll_sentences:

        sentence_txt = ' '.join([a['form'] for a in sentence.tokens])
        pieces = sp.EncodeAsPieces(sentence_txt)

        # get tokens to pieces correspondances

        pieces = [a for a in pieces if a != '▁']
        correspondances = _get_correspondances(pieces, sentence)  # i.e. [([0], [0]), ([1], [1]), ([2], [2]), ([3, 4], [3]), ([5], [4])]

        tokens_to_add = []
        tokens_to_remove = []

        for piece_ids, token_ids in correspondances:

            original_token = sentence[token_ids[0]]

            # save original token to remove it later and create new tokens for each one of the pieces
            if len(piece_ids) > 1:

                tokens_to_remove.append(original_token)

                # increase ids and heads in sentence

                num_new_nodes = len(piece_ids) - 1
                origin_id = original_token['id']
                _increment_ids(sentence, tokens_to_add, num_new_nodes, origin_id)

                # add new tokens to new tokens list

                for i, piece_id in enumerate(piece_ids):

                    id = origin_id + i

                    original_form = pieces[piece_id].replace('▁', '')
                    form = original_form if i == 0 else '@@@%s' % original_form
                    lemma = form

                    upostag = original_token['upostag'] if i == 0 else 'BPE'
                    xpostag = original_token['xpostag'] if i == 0 else 'BPE'
                    feats = original_token['feats'] if i == 0 else None
                    head = original_token['head'] if i == 0 else origin_id
                    deprel = original_token['deprel'] if i == 0 else 'BPE'
                    deps = original_token['deps'] if i == 0 else None
                    misc = original_token['misc'] if i == 0 else None
                    new_token = OrderedDict(
                        [('id', id), ('form', form), ('lemma', lemma), ('upostag', upostag), ('xpostag', xpostag),
                         ('feats', feats), ('head', head), ('deprel', deprel), ('deps', deps), ('misc', misc)])
                    tokens_to_add.append(new_token)

        # add new tokens to sentence
        for new_token in tokens_to_add:
            sentence.append(new_token)

        # remove tokens that have been split from sentence
        for token in tokens_to_remove:
            sentence.remove(token)

        sentence.sort(key=lambda x: x['id'], reverse=False)

        # remove sentence metadata to avoid comments on conll file
        sentence.metadata = None

        output_file.write(sentence.serialize())


if __name__ == "__main__":
    """ 
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(asctime)s:\t%(message)s")

    # 0. Generate txt version of PTB

    if False:
        conllu_to_text('/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/dev.gold.conll', '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/dev.gold.txt')
        conllu_to_text('/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/test.gold.conll', '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/test.gold.txt')
        conllu_to_text('/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.conll', '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.txt')


    # 1. train bpe model with [raw 1b text training set + raw penn training set] 
    #    as input, 80k vocab --> bpe.model; bpe.vocab

    input_files = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.txt,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00000-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00001-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00002-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00003-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00004-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00005-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00006-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00007-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00008-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00009-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00010-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00011-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00012-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00013-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00014-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00015-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00016-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00017-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00018-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00019-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00020-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00021-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00022-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00023-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00024-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00025-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00026-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00027-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00028-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00029-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00030-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00031-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00032-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00033-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00034-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00035-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00036-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00037-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00038-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00039-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00040-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00041-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00042-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00043-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00044-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00045-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00046-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00047-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00048-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00049-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00050-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00051-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00052-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00053-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00054-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00055-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00056-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00057-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00058-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00059-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00060-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00061-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00062-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00063-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00064-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00065-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00066-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00067-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00068-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00069-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00070-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00071-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00072-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00073-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00074-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00075-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00076-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00077-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00078-of-00100,/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00079-of-00100'
    input_files_hpc = '/homedtic/lperez/UniParse/datasets/PTB_SD_3_3_0/train.gold.txt,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00001-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00002-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00003-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00004-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00005-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00006-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00007-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00008-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00009-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00010-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00011-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00012-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00013-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00014-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00015-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00016-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00017-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00018-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00019-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00020-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00021-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00022-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00023-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00024-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00025-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00026-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00027-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00028-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00029-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00030-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00031-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00032-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00033-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00034-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00035-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00036-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00037-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00038-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00039-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00040-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00041-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00042-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00043-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00044-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00045-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00046-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00047-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00048-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00049-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00050-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00051-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00052-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00053-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00054-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00055-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00056-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00057-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00058-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00059-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00060-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00061-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00062-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00063-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00064-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00065-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00066-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00067-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00068-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00069-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00070-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00071-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00072-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00073-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00074-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00075-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00076-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00077-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00078-of-00100,/homedtic/lperez/datasets/1-billion-word-language-modeling-benchmark-r13output/text/training-monolingual.tokenized.shuffled/news.en-00079-of-00100'
    vocab_size = '80000'
    model_type = 'bpe'
    model_prefix = 'bpe'
    character_coverage = '1.0'

    if False:
        spm.SentencePieceTrainer.Train('--input=%s --model_prefix=%s --vocab_size=%s  --character_coverage=%s --model_type=%s' % (input_files_hpc, model_prefix, vocab_size, character_coverage, model_type))


    # dummy testing reconstruction

    if False:
        conll_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/prova.conll'
        output_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/prova.bpe.conll'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)


    # 2a. reconstruct penn dependency trees with bpe model --> [PTB bpe]

    if False:
        conll_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/dev.gold.conll'
        output_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/dev.gold.bpe.conll'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)

        conll_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.conll'
        output_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.bpe.conll'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)

        conll_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/test.gold.conll'
        output_file = '/home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/test.gold.bpe.conll'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)


    # 2b. reconstruct 1B dependency trees with bpe model --> [1B bpe]

    if True:
        # conll_file = '/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_normal/1B_test.conllu'
        # output_file = '/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe/1B_test.bpe.conllu'
        # reconstruct_dependency_trees(conll_file, model_prefix, output_file)

        conll_file = '/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_normal/1B_dev.conllu'
        output_file = '/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe/1B_dev.bpe.conllu'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)

        conll_file = '/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_normal/1B_train.conllu'
        output_file = '/home/lpmayos/code/datasets/1-billion-word-language-modeling-benchmark-r13output/conll_bpe/1B_train.bpe.conllu'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)


    # 3a. train kiperwasser parser with [PTB bpe] --> model_bpe.model & vocab_bpe.pkl

    # python kiperwasser_main.py --results_folder /home/lpmayos/code/UniParse/saved_models/kiperwasser_en_ptb_BPE --logging_file logging.log --do_training True --train_file /home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/train.gold.bpe.conll --dev_file /home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/dev.gold.bpe.conll --test_file /home/lpmayos/code/UniParse/datasets/PTB_SD_3_3_0/test.gold.bpe.conll --output_file output_ptb_bpe.output --model_file model_bpe.model --vocab_file vocab_bpe.pkl


    # 3b. execute bpe encoding on 1B dataset --> text files with @@@x marking the bpe tokens
    #   --> /home/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text_bpe/heldout-monolingual.tokenized.shuffled/...
    #   --> /home/code/datasets/1-billion-word-language-modeling-benchmark-r13output/text_bpe/training-monolingual.tokenized.shuffled/...

    # bash bpe_1B.sh


    # can we parse text files with kiperwasser?? not directly, as it expects a conll file
    # 4. convert [1B bpe text] into [1B bpe conll] using corenlp (parse_1b_with_corenlp.py)
    #    $ python parse_1b_with_corenlp.py --input_dir ~/code/datasets/1-billion-word-language-modeling-benchmark-r13output/ --heldout_folder heldout-monolingual.tokenized.shuffled --training_folder training-monolingual.tokenized.shuffled
    #       --> we get tokenization and POS tagging! (along with parsing, that we don't need and we won't use)
    #       --> head and deprel do not matter, bc when parsing with kiperwasser they will not be taken into account
    #           (I tested this by parsing the same file with different head-deprel, obtaining the same results)

    # 5. use kiperwasser + model_bpe.model to parse [1B bpe conll] --> a lot of parsed conll files


    # 6. generate 1B_bpe_dev.conllu, 1B_bpe_test.conllu, 1B_bpe_train.conllu
    # (using create_1b_train_dev_test_splits function from parse_1b_with_corenlp.py)

    # 7. train kiperwasser with 1B_bpe_dev.conllu, 1B_bpe_test.conllu, 1B_bpe_train.conllu


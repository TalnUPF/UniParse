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
    """ adds extra conll file for tokens that have been splitted by sentencepiece
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

            if piece == token:
                correspondance = ([i_pieces], [i_sentence])

            elif len(token) > len(piece): 
                # add pieces until they are the same
                correspondance = ([i_pieces], [i_sentence])
                while piece != token:
                    i_pieces += 1
                    new_piece = pieces[i_pieces].replace('▁', ' ')
                    piece += new_piece
                    correspondance[0].append(i_pieces)

            else:
                # add tokens until they are the same
                correspondance = ([i_pieces], [i_sentence])
                while piece != token:
                    i_sentence += 1
                    new_token = sentence[i_sentence]
                    token += new_token
                    correspondance[1].append(i_sentence)

            i_pieces += 1
            i_sentence += 1
            correspondances.append(correspondance)

        return correspondances


    def _increment_ids(sentence, increment, origin):
        """ increments in 'increment' the id of all tokens with id > origin;
        increments in 'increment' the head of all tokens with head > origin;
        """
        for token in sentence:
            if token['id'] > origin:
                token['id'] += increment
            if token['head'] > origin + 1:
                token['head'] += increment

    def _add_new_pieces_as_tokens(piece_ids, original_token, pieces, new_sentence):
        """
        """
        piece_initial_id = piece_ids[0] + 1

        for i, piece_id in enumerate(piece_ids):
            id = piece_initial_id + i
            form = pieces[piece_id].replace('▁', '')
            lemma = form
            upostag = original_token['upostag'] if i == 0 else None
            xpostag = original_token['xpostag'] if i == 0 else None
            feats = original_token['feats'] if i == 0 else None
            head = original_token['head'] if i == 0 else piece_initial_id
            deprel = original_token['deprel'] if i == 0 else 'PIECED'
            deps = original_token['deps'] if i == 0 else None
            misc = original_token['misc'] if i == 0 else None
            new_token = OrderedDict([('id', id), ('form', form), ('lemma', lemma), ('upostag', upostag), ('xpostag', xpostag), ('feats', feats), ('head', head), ('deprel', deprel), ('deps', deps), ('misc', misc)])
            new_sentence.append(new_token)

    conll_file = open(conll_file_path, "r", encoding="utf-8")
    output_file = open(output_file_path, "w", encoding="utf-8")

    conll_sentences = parse_incr(conll_file)

    sp = spm.SentencePieceProcessor()                                                                                                                                                                            
    sp.Load('%s.model' % model_prefix)

    for sentence in conll_sentences:

        sentence_txt = ' '.join([a['form'] for a in sentence.tokens])
        pieces = sp.EncodeAsPieces(sentence_txt)

        # sentence_txt  --> "In the plan 's first stage , the Palestinians were to dismantle armed groups ."
        # pieces        --> ['▁In', '▁the', '▁plan', "▁'", 's', '▁first', '▁stage', '▁,', '▁the', '▁Palestinians', '▁were', '▁to', '▁dismantle', '▁armed', '▁groups', '▁.']
        # sentence      --> sentence<In, the, plan, 's, first, stage, ,, the, Palestinians, were, to, dismantle, armed, groups, .>
        
        correspondances = _get_correspondances(pieces, sentence)  # i.e. [([0], [0]), ([1], [1]), ([2], [2]), ([3, 4], [3]), ([5], [4])]
        new_sentence = TokenList([])

        for piece_ids, token_ids in correspondances:

            original_token = sentence[token_ids[0]]

            if len(piece_ids) == len(token_ids) == 1: 
                new_sentence.append(original_token)

            elif len(piece_ids) > 1 and len(token_ids) == 1: 
                num_new_nodes = len(piece_ids) - 1
                _increment_ids(sentence, num_new_nodes, piece_ids[0])
                _add_new_pieces_as_tokens(piece_ids, original_token, pieces, new_sentence)

            elif len(piece_ids) == 1 and len(token_ids) > 1:
                logging.error('More tokens than pieces found for sentence %s. Debug needed!' % sentence_txt)

            else:
                logging.error('Ill correspondance found for sentence %s. Debug needed!' % sentence_txt)

        print(new_sentence.serialize())
        output_file.write(new_sentence.serialize())


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
    model_prefix = 'm'
    character_coverage = '1.0'

    if False:
        spm.SentencePieceTrainer.Train('--input=%s --model_prefix=%s --vocab_size=%s  --character_coverage=%s --model_type=%s' % (input_files_hpc, model_prefix, vocab_size, character_coverage, model_type))


    # 2. encode penn (all files) using bpe.model (bash)

    # spm_encode --model=<model_file> --output_format=piece < input > output


    # 3. reconstruct penn dependency trees: iterate simultaneously over penn 
    #    enocoded files and conll files, splitting lines when necessary (if 
    #    a word is divided in two, first part is head and rest is dependant).
    #    --> penn_train_pieced.conll, penn_dev_pieced.conll and 
    #        penn_test_pieced.conll

    if False:
        conll_file = 'babau4.conllu'
        output_file = 'babau4_pieced.conllu'
        reconstruct_dependency_trees(conll_file, model_prefix, output_file)


    # 4. train kiperwasser/StanfordCoreNLP parser with penn_train_pieced.conll, 
    #    penn_dev_pieced.conll and penn_test_pieced.conll --> model1_pieced

    # 5. use kiperwasser/StanfordCoreNLP + model1_pieced to parse 
    #    1B input_pieced.txt[1b] --> 1b_train_pieced.conll, 1b_dev_pieced.conll, 
    #    1b_test_pieced.conll

    # 6. train kiperwasser parser with 1b_train_pieced.conll, 
    #    1b_dev_pieced.conll, 1b_test_pieced --> model3

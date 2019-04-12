from uniparse import Vocabulary

# script to check how big is the generated vocabulary

vocab_file = '/home/lpmayos/code/UniParse/saved_models/kiperwasser_en_ptb_pieced/vocab_ptb_pieced.pkl'
vocab = Vocabulary()
vocab.load(vocab_file)
import ipdb; ipdb.set_trace()
print('babau')
import torch

sourceFileName = "en_bg_data/train.bg"
targetFileName = "en_bg_data/train.en"
sourceDevFileName = "en_bg_data/dev.bg"
targetDevFileName = "en_bg_data/dev.en"

corpusFileName = "corpusData"
wordsFileName = "wordsData"
modelFileName = "NMTmodel"

device = torch.device("cuda:0")
# device = torch.device("cpu")

d_model = 512
n_heads = 8
n_layers = 3
dropout = 0.1

learning_rate = 5e-4
weight_decay = 1e-4
batchSize = 64
clip_grad = 1.0

maxEpochs = 20
log_every = 1000
test_every = 1000

startToken = "<S>"
startTokenIdx = 0

endToken = "</S>"
endTokenIdx = 1

unkToken = "<UNK>"
unkTokenIdx = 2

padToken = "<PAD>"
padTokenIdx = 3

transToken = "<TRANS>"
transTokenIdx = 4


# BPE settings
use_bpe = True
bpe_merges = 30000
bpe_codes_file = "bpe_codes"

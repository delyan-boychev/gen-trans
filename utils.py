#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
import numpy as np
import random
import os
import nltk
from nltk.translate.bleu_score import corpus_bleu
from bpe_tokenization import *
from common import progressBar

nltk.download("punkt")


def setSeed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def readCorpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print("Loading file:", fileName)
    return [nltk.word_tokenize(line) for line in open(fileName)]


def getDictionary(
    corpus, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold=2
):
    dictionary = {}
    for s in corpus:
        for w in s:
            if w in dictionary:
                dictionary[w] += 1
            else:
                dictionary[w] = 1

    # Премахваме за всеки случай специалните токени, ако съществуват в корпуса (най-вероятно не, но за сигурност)
    special_tokens = {startToken, endToken, unkToken, padToken, transToken}
    for st in special_tokens:
        if st in dictionary:
            del dictionary[st]

    words = [startToken, endToken, unkToken, padToken, transToken] + [
        w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold
    ]
    return {w: i for i, w in enumerate(words)}


def prepareData(
    sourceFileName,
    targetFileName,
    sourceDevFileName,
    targetDevFileName,
    startToken,
    endToken,
    unkToken,
    padToken,
    transToken,
    useBpe=False,
    bpeMerges=30000,
    bpeCodesFile="bpe_codes",
):

    sourceCorpus = readCorpus(sourceFileName)
    targetCorpus = readCorpus(targetFileName)
    if useBpe:
        # Ако правилата вече са записани, не ги преизчисляваме.
        if os.path.exists(bpeCodesFile):
            codes = loadBpeCodes(bpeCodesFile)
        else:
            codes = learnBpe(sourceCorpus + targetCorpus, bpeMerges)
            saveBpeCodes(codes, bpeCodesFile)
        # Прилагаме BPE върху training set-а
        sourceCorpus = applyBpe(sourceCorpus, codes)
        targetCorpus = applyBpe(targetCorpus, codes)
    word2ind = getDictionary(
        sourceCorpus + targetCorpus,
        startToken,
        endToken,
        unkToken,
        padToken,
        transToken,
    )

    trainCorpus = [
        [startToken] + s + [transToken] + t + [endToken]
        for (s, t) in zip(sourceCorpus, targetCorpus)
    ]

    sourceDev = readCorpus(sourceDevFileName)
    targetDev = readCorpus(targetDevFileName)
    if useBpe:
        sourceDev = applyBpe(sourceDev, codes)
        targetDev = applyBpe(targetDev, codes)

    devCorpus = [
        [startToken] + s + [transToken] + t + [endToken]
        for (s, t) in zip(sourceDev, targetDev)
    ]

    print("Corpus loading completed.")
    return trainCorpus, devCorpus, word2ind

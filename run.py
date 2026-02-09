#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time
import json

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *


def perplexity(nmt, test, batchSize):
    testSize = len(test)
    H = 0.0
    c = 0
    for b in range(0, testSize, batchSize):
        batch = test[b : min(b + batchSize, testSize)]
        l = sum(len(s) - 1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * nmt(batch)
    return math.exp(H / c)


if len(sys.argv) > 1 and sys.argv[1] == "prepare":
    trainCorpus, devCorpus, word2ind = utils.prepareData(
        sourceFileName,
        targetFileName,
        sourceDevFileName,
        targetDevFileName,
        startToken,
        endToken,
        unkToken,
        padToken,
        transToken,
        use_bpe,
        bpe_merges,
        bpe_codes_file,
    )
    # Convert to indices for storage (legacy compatibility)
    trainCorpus = [[word2ind.get(w, unkTokenIdx) for w in s] for s in trainCorpus]
    devCorpus = [[word2ind.get(w, unkTokenIdx) for w in s] for s in devCorpus]
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, "wb"))
    pickle.dump(word2ind, open(wordsFileName, "wb"))
    print("Data prepared.")

if len(sys.argv) > 1 and (sys.argv[1] == "train" or sys.argv[1] == "extratrain"):
    utils.setSeed()
    (trainCorpus, devCorpus) = pickle.load(open(corpusFileName, "rb"))
    word2ind = pickle.load(open(wordsFileName, "rb"))

    # Helper to convert indices back to words for the new model if needed,
    # but here we pass indices to perplexity which expects indices.
    # Wait, perplexity function calls nmt(batch). nmt forward expects indices.
    # So devCorpus (indices) is correct.

    nmt = model.LanguageModel(len(word2ind), d_model, n_heads, n_layers, dropout).to(
        device
    )

    optimizer = torch.optim.Adam(
        nmt.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_min = 1e-6
    steps_per_epoch = math.ceil(len(trainCorpus) / batchSize)
    total_steps = maxEpochs * steps_per_epoch
    warmup_steps = max(1, int(0.2 * steps_per_epoch))

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=lr_min / learning_rate,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=lr_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    metrics_log = []

    if sys.argv[1] == "extratrain":
        nmt.load(modelFileName)
        (iter, bestPerplexity, learning_rate, osd) = torch.load(
            modelFileName + ".optim"
        )
        optimizer.load_state_dict(osd)
        # Load metrics log if exists
        try:
            with open("training_metrics.json", "r") as f:
                metrics_log = json.load(f)
        except FileNotFoundError:
            pass

        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
            last_epoch=iter - 1,
        )
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype="int32")
    nmt.train()
    beginTime = time.time()

    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()

        total_epoch_loss_sum = 0.0
        total_epoch_tokens = 0
        epoch_grad_norm = 0.0
        batches_in_epoch = 0

        for b in range(0, len(idx), batchSize):
            iter += 1
            batch = [trainCorpus[i] for i in idx[b : min(b + batchSize, len(idx))]]

            batch_tokens = sum(len(s) - 1 for s in batch)
            words += batch_tokens

            H = nmt(batch)

            # H is average loss per token. Multiply by tokens to get total sum.
            total_epoch_loss_sum += H.item() * batch_tokens
            total_epoch_tokens += batch_tokens

            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)

            epoch_grad_norm += grad_norm.item()
            batches_in_epoch += 1

            optimizer.step()
            scheduler.step()

            if iter % log_every == 0:
                print(
                    "Iteration:",
                    iter,
                    "Epoch:",
                    epoch + 1,
                    "/",
                    maxEpochs,
                    ", Batch:",
                    b // batchSize + 1,
                    "/",
                    len(idx) // batchSize + 1,
                    ", loss: ",
                    H.item(),
                    "words/sec:",
                    words / (time.time() - trainTime),
                    "time elapsed:",
                    (time.time() - beginTime),
                )
                trainTime = time.time()
                words = 0

            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpus, batchSize)
                nmt.train()
                print("Current model perplexity: ", currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print("Saving new best model.")
                    nmt.save(modelFileName)
                    torch.save(
                        (iter, bestPerplexity, learning_rate, optimizer.state_dict()),
                        modelFileName + ".optim",
                    )

        # End of epoch logging
        nmt.eval()
        val_perplexity = perplexity(nmt, devCorpus, batchSize)
        nmt.train()

        # Calculate precise weighted average loss
        avg_epoch_loss = (
            total_epoch_loss_sum / total_epoch_tokens if total_epoch_tokens > 0 else 0.0
        )
        avg_grad_norm = epoch_grad_norm / batches_in_epoch
        current_lr = scheduler.get_last_lr()[0]

        epoch_metrics = {
            "epoch": epoch + 1,
            "val_perplexity": val_perplexity,
            "avg_train_loss": avg_epoch_loss,
            "avg_grad_norm": avg_grad_norm,
            "learning_rate": current_lr,
        }
        metrics_log.append(epoch_metrics)

        with open("training_metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=4)

        print(f"Epoch {epoch+1} finished. Metrics saved.")

    print("reached maximum number of epochs!")
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpus, batchSize)
    print("Last model perplexity: ", currentPerplexity)

    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print("Saving last model.")
        nmt.save(modelFileName)
        torch.save(
            (iter, bestPerplexity, learning_rate, optimizer.state_dict()),
            modelFileName + ".optim",
        )

if len(sys.argv) > 3 and sys.argv[1] == "perplexity":
    word2ind = pickle.load(open(wordsFileName, "rb"))

    nmt = model.LanguageModel(len(word2ind), d_model, n_heads, n_layers, dropout).to(
        device
    )
    nmt.load(modelFileName)

    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    if use_bpe:
        codes = utils.loadBpeCodes(bpe_codes_file)
        sourceTest = utils.applyBpe(sourceTest, codes)
        targetTest = utils.applyBpe(targetTest, codes)
    test = [
        [startToken] + s + [transToken] + t + [endToken]
        for (s, t) in zip(sourceTest, targetTest)
    ]
    test = [[word2ind.get(w, unkTokenIdx) for w in s] for s in test]

    nmt.eval()
    print("Model perplexity: ", perplexity(nmt, test, batchSize))

if len(sys.argv) > 3 and sys.argv[1] == "translate":
    word2ind = pickle.load(open(wordsFileName, "rb"))
    words = list(word2ind)

    sourceTest = utils.readCorpus(sys.argv[2])
    if use_bpe:
        codes = utils.loadBpeCodes(bpe_codes_file)
        sourceTest = utils.applyBpe(sourceTest, codes)
    test = [[startToken] + s + [transToken] for s in sourceTest]
    test = [[word2ind.get(w, unkTokenIdx) for w in s] for s in test]

    nmt = model.LanguageModel(len(word2ind), d_model, n_heads, n_layers, dropout).to(
        device
    )
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3], "w")
    pb = utils.progressBar()
    pb.start(len(test))
    for s in test:
        r = nmt.generate(s)
        st = r.index(transTokenIdx)
        result = [words[i] for i in r[st + 1 : -1]]
        if use_bpe:
            result = utils.decodeBpe(result)
        file.write(" ".join(result) + "\n")
        pb.tick()
    pb.stop()

if len(sys.argv) > 2 and sys.argv[1] == "generate":
    word2ind = pickle.load(open(wordsFileName, "rb"))
    words = list(word2ind)

    test = sys.argv[2].split()
    if use_bpe:
        codes = utils.loadBpeCodes(bpe_codes_file)
        test = utils.applyBpe([test], codes)[0]
    test = [word2ind.get(w, unkTokenIdx) for w in test]

    nmt = model.LanguageModel(len(word2ind), d_model, n_heads, n_layers, dropout).to(
        device
    )
    nmt.load(modelFileName)

    nmt.eval()
    r = nmt.generate(test)
    result = [words[i] for i in r]
    if use_bpe:
        result = utils.decodeBpe(result)
    print(" ".join(result) + "\n")

if len(sys.argv) > 3 and sys.argv[1] == "bleu":
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print("Corpus BLEU: ", (bleu_score * 100))

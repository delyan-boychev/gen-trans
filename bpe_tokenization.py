import heapq
from collections import Counter
from common import progressBar


def saveBpeCodes(codes, fileName):
    # Записва научените BPE правила за сливане в текстов файл
    # Всеки ред съдържа двата елемента за слизване
    with open(fileName, "w") as f:
        for a, b in codes:
            f.write(a + " " + b + "\n")


def loadBpeCodes(fileName):
    # Зарежда BPE правилата от файл
    # Връща списък от двойки, подредени по ред на сливане и лексикографски
    codes = []
    with open(fileName, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                codes.append((parts[0], parts[1]))
    return codes


def _merge_pair(word, pair):
    # Слива всички срещания на дадена двойка символи в дума
    merged = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def learnBpe(corpus, numMerges):
    # Обучава BPE върху корпуса, като извършва numMerges сливания
    word_counts = Counter()
    for sent in corpus:
        for w in sent:
            word_counts[w] += 1

    # Представяме всяка дума като символи + край на дума </w>
    word_freq = {tuple(list(w) + ["</w>"]): c for w, c in word_counts.items()}
    word_pairs = {}
    pair_counts = Counter()
    pair_to_words = {}

    # Пазим наредените двойки от (-честота, двойка от токени)
    pq = []

    for word, freq in word_freq.items():
        # Инициализира броя на всички двойки в думата
        # pair_to_words пази кои думи съдържат дадена двойка
        pairs = []
        for i in range(len(word) - 1):
            p = (word[i], word[i + 1])
            pairs.append(p)
            pair_counts[p] += freq
            pair_to_words.setdefault(p, set()).add(word)
        word_pairs[word] = pairs

    for p, count in pair_counts.items():
        heapq.heappush(pq, (-count, p))

    merges = []
    pb = progressBar()
    pb.start(numMerges)

    for _ in range(numMerges):
        # Избира най-честата двойка и обновява само засегнатите думи
        best = None
        while pq:
            # Вадим най-честото (най-малкото отрицателно число)
            neg_count, pair = heapq.heappop(pq)

            # Ако pair_counts[pair] се е променило, значи този запис в heap-а е стар (понеже не ги трием, защото е доста бавно)
            if pair_counts[pair] == -neg_count:
                best = pair
                break
            # Ако не съвпада, просто цикълът се върти пак и вади следващия

        if not best:  # Ако хийпът е празен или няма валидни
            break

        merges.append(best)
        affected = pair_to_words.get(best, set())

        # Обновява само думите, които съдържат избраната двойка
        for word in list(affected):
            freq = word_freq.pop(word)
            old_pairs = word_pairs.pop(word)
            # Премахва старите двойки за думата
            for p in old_pairs:
                pair_counts[p] -= freq
                if pair_counts[p] > 0:
                    heapq.heappush(pq, (-pair_counts[p], p))
                elif pair_counts[p] <= 0:
                    del pair_counts[p]
                ws = pair_to_words.get(p)
                if ws is not None:
                    ws.discard(word)
                    if not ws:
                        pair_to_words.pop(p, None)

            # Слива най-добрата двойка в думата и добавя новите двойки
            new_word = _merge_pair(word, best)
            word_freq[new_word] = word_freq.get(new_word, 0) + freq

            new_pairs = []
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                new_pairs.append(p)
                pair_counts[p] += freq
                pair_to_words.setdefault(p, set()).add(new_word)
                # Винаги, когато променим честотата слагаме в хийпа
                heapq.heappush(pq, (-pair_counts[p], p))
            word_pairs[new_word] = new_pairs

        pb.tick()

    pb.stop()
    return merges


def _apply_bpe_to_word(word, codes_dict, cache):
    # Проверка в кеша преди да я представим като отделни токени
    if word in cache:
        return cache[word]

    # Специалните токени
    if word.startswith("<") and word.endswith(">"):
        return [word]

    symbols = list(word) + ["</w>"]

    while len(symbols) > 1:
        # Намираме всички налични двойки
        best_pair = None
        min_rank = float("inf")

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = codes_dict.get(pair)
            if rank is not None and rank < min_rank:
                min_rank = rank
                best_pair = pair

        if best_pair is None:
            break

        symbols = list(_merge_pair(tuple(symbols), best_pair))

    output = []

    for i, sym in enumerate(symbols):
        if sym == "</w>":
            continue

        clean_sym = sym.replace("</w>", "")

        # Логика за @@
        if i == len(symbols) - 1:  # Последен символ
            output.append(clean_sym)
        elif (
            i + 1 < len(symbols) and symbols[i + 1] == "</w>"
        ):  # Предпоследен, следван от маркер
            output.append(clean_sym)
        else:
            output.append(clean_sym + "@@")

    # Записваме в кеша преди да върнем кодираната дума
    cache[word] = output
    return output


def applyBpe(corpus, codes):
    codes_dict = {pair: i for i, pair in enumerate(codes)}

    # Създаване на локален кеш
    word_cache = {}

    new_corpus = []
    for sent in corpus:
        new_sent = []
        for w in sent:
            # Подаваме кеша надолу
            new_sent.extend(_apply_bpe_to_word(w, codes_dict, word_cache))
        new_corpus.append(new_sent)
    return new_corpus


def decodeBpe(tokens):
    # Възстановява думите от BPE токени с маркер @@
    # Слепя последователните поддуми до цяла дума
    words = []
    current = ""
    for tok in tokens:
        if tok.endswith("@@"):
            current += tok[:-2]
        else:
            current += tok
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words

import torch
import torch.nn as nn
import math
from parameters import *


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(1, max_len, d_model)

        position = torch.arange(max_len).unsqueeze(0).unsqueeze(2)
        div_term = (
            (10000.0 ** (torch.arange(0, d_model, 2) / d_model))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = (
            x + self.pe[:, : x.shape[1], :]
        )  # x.shape = (batch_size, seq_len, embedding_dim)
        return x


class MultiHeadAttn(torch.nn.Module):
    def __init__(self, n_head, d_model, d_keys, d_values, dropout=0.1):
        super(MultiHeadAttn, self).__init__()

        self.n_head, self.d_model, self.d_keys, self.d_values = (
            n_head,
            d_model,
            d_keys,
            d_values,
        )
        self.scale = 1 / (d_keys**0.5)

        self.Wq_net = torch.nn.Linear(d_model, n_head * d_keys)
        self.Wk_net = torch.nn.Linear(d_model, n_head * d_keys)
        self.Wv_net = torch.nn.Linear(d_model, n_head * d_values)
        self.Wo_net = torch.nn.Linear(n_head * d_values, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, input, attn_mask=None, cache=None):

        n_head, d_model, d_keys, d_values = (
            self.n_head,
            self.d_model,
            self.d_keys,
            self.d_values,
        )

        batch_size = input.shape[0]
        seq_len = input.shape[1]
        # input.shape = (batch_size, seq_len, d_model)
        head_q = self.Wq_net(input)
        head_k = self.Wk_net(input)
        head_v = self.Wv_net(input)
        # {head_q,head_k}.shape = (batch_size, seq_len, n_head * d_keys)
        # head_v.shape = (batch_size, seq_len, n_head * d_values)

        q = head_q.view(batch_size, seq_len, n_head, d_keys).transpose(1, 2)
        k = head_k.view(batch_size, seq_len, n_head, d_keys).permute(0, 2, 3, 1)
        v = head_v.view(batch_size, seq_len, n_head, d_values).transpose(1, 2)
        # q.shape = (batch_size, n_head, seq_len, d_keys)
        # k.shape = (batch_size, n_head, d_keys, seq_len)
        # v.shape = (batch_size, n_head, seq_len, d_values)

        if cache is not None:
            # Кешът е (past_k, past_v)
            past_k, past_v = cache
            # Конкатенираме по измерението, което е дължината (т.е. да добавим скритите състояния за предходните токени)
            # k: (batch, n_head, d_keys, seq_len) -> dim 3
            # v: (batch, n_head, seq_len, d_values) -> dim 2
            k = torch.cat([past_k, k], dim=3)
            v = torch.cat([past_v, v], dim=2)

        new_cache = (k, v)

        attn_score = torch.matmul(q, k)
        # attn_score.shape = (batch_size, n_head, seq_len, total_seq_len)

        attn_score = attn_score * self.scale

        if attn_mask is not None:
            # attn_mask = (seq_len, total_seq_len)
            attn_score = attn_score.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(1), -float("inf")
            )

        attn_prob = torch.nn.functional.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)

        # attn_prob.shape = (batch_size, n_head, seq_len, total_seq_len)
        attn_vec = torch.matmul(attn_prob, v)
        # attn_vec.shape = (batch_size, n_head, seq_len, d_values)

        attn_vec = attn_vec.transpose(1, 2).flatten(2, 3)
        # attn_vec.shape = (batch_size, seq_len, n_head * d_values)

        # linear projection
        attn_out = self.Wo_net(attn_vec)
        # attn_out = (batch_size, seq_len, d_model)
        return attn_out, new_cache


class Transformer_cell(torch.nn.Module):

    def __init__(self, n_head, d_model, d_keys, d_values, d_ff, dropout):

        super().__init__()
        self.MHA = MultiHeadAttn(n_head, d_model, d_keys, d_values, dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(d_model)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.W1 = torch.nn.Linear(d_model, d_ff)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.W2 = torch.nn.Linear(d_ff, d_model)
        self.layer_norm_2 = torch.nn.LayerNorm(d_model)
        self.dropout_3 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None, cache=None):

        # x.shape = (batch_size, seq_len, d_model)
        # mask.shape = (seq_len, seq_len)
        z1, new_cache = self.MHA(x, mask, cache)
        z2 = self.layer_norm_1(x + self.dropout_1(z1))

        z3 = self.W2(
            self.dropout_2(torch.nn.functional.relu(self.W1(z2)))
        )  ## Feed Forward
        y = self.layer_norm_2(z2 + self.dropout_3(z3))

        return y, new_cache


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout):
        super(LanguageModel, self).__init__()
        self.d_model = d_model
        self.unkTokenIdx = unkTokenIdx
        self.padTokenIdx = padTokenIdx
        self.endTokenIdx = endTokenIdx

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Стакваме словете като лист от nn.Module
        self.layers = nn.ModuleList(
            [
                Transformer_cell(
                    nhead,  # брой на главите
                    d_model,
                    d_model
                    // nhead,  # key се разбива на броя глави и се смята за всяка глава поотделно
                    d_model // nhead,  # за query аналогично
                    d_model * 4,  # ff размерност е 4*d_model
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Максимална дължина на поредицата
        maxlen = 5000
        pos = torch.arange(maxlen)
        mask = pos.unsqueeze(0) > pos.unsqueeze(1)
        self.register_buffer("mask", mask)
        self.embed_scaling = math.sqrt(self.d_model)
        self.projection = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        # Инициализация на параметрите (заимствано от pytorch имплементацията на transformer)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in source]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)

    def _forward_step(self, x, mask=None, cache=None):
        seq_len = x.shape[1]

        E = self.embed(x) * self.embed_scaling

        # Позиционни емебедигни, първо проверяваме дали имаме кеш и трябва ли да ги смятаме
        if cache is None or cache[0] is None:
            # Training or Initial Prompt: Standard PE (0 to seq_len)
            input = self.dropout1(self.pos_embed(E))
        else:
            # Генериране
            # cache[0] е наредена двойка (k, v). к е наредената четворка (Batch, Heads, PastLen, D_head), от която вземаме дължината
            past_len = cache[0][0].shape[3]
            pe_slice = self.pos_embed.pe[:, past_len : past_len + seq_len, :]
            input = self.dropout1(E + pe_slice)

        # Forward pass през transfomer слоевете
        out = input
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            out, c = layer(out, mask, layer_cache)
            new_cache.append(c)

        logits = self.projection(out)
        return logits, new_cache

    def forward(self, source):
        X = self.preparePaddedBatch(source)
        # X: (N, L)
        src = X[:, :-1]
        tgt = X[:, 1:]

        seq_len = src.shape[1]
        # Отрязваме маската за текущата максимална дължина

        batch_mask = self.mask[:seq_len, :seq_len]

        logits, _ = self._forward_step(src, mask=batch_mask, cache=None)
        logits_flat = logits.flatten(0, 1)
        tgt_flat = tgt.flatten(0, 1)
        # Смятаме грешка изключвайки допълващите токени
        H = torch.nn.functional.cross_entropy(
            logits_flat,
            tgt_flat,
            ignore_index=self.padTokenIdx,
            label_smoothing=0.1 if self.training else 0.0,
        )
        return H

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    @torch.no_grad()
    def generate(self, source, limit=1000):
        self.eval()
        device = next(self.parameters()).device
        seq = source.copy()

        # Задаваме начален кеш, който ще пазим за всички слоеве
        # Това ще бъдат всички hidden state-ове, които имаме до момента изчислени, за да не преизчисляваме
        cache = [None] * len(self.layers)

        if len(seq) > 0:
            # Инициализираме си тензор с индексите на токените
            X = torch.tensor([seq], dtype=torch.long, device=device)
            # Маската отново е аналогична на forward
            seq_len = X.shape[1]
            batch_mask = self.mask[:seq_len, :seq_len]
            # Даваме кешът, който сме запазили като аргумент и получаваме новия
            logits, cache = self._forward_step(X, mask=batch_mask, cache=cache)

            # Предскзваме следващият токен, махайки unknown и pad токените
            last_logits = logits[:, -1, :]
            last_logits[0, [self.unkTokenIdx, self.padTokenIdx]] = -float("inf")
            next_token = torch.argmax(last_logits, dim=-1).item()

            seq.append(next_token)
            if next_token == self.endTokenIdx:
                return seq

        # След като вече имаме кеш за всички начални можем да продължим
        for _ in range(limit):
            # Input is just the last generated token
            X = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)

            # Отново смятаме и актуализираме кеша
            logits, cache = self._forward_step(X, mask=None, cache=cache)

            last_logits = logits[:, -1, :]
            last_logits[0, [self.unkTokenIdx, self.padTokenIdx]] = -float("inf")
            # Аналогично на преди
            next_token = torch.argmax(last_logits, dim=-1).item()
            seq.append(next_token)

            if next_token == self.endTokenIdx:
                break

        return seq

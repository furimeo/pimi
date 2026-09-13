import json
import re

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('\xa1'), ord('\xac')+1)) + list(range(ord('\xae'), ord('\xff')+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(i) for i in cs]))

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

vocab = json.load(open('models/vocab.json', encoding='utf-8'))
merges = [line.strip().split() for line in open('models/merges.txt', encoding='utf-8').read().splitlines()[1:50001] if line.strip()]
bpe_ranks = {tuple(m): i for i, m in enumerate(merges)}

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def bpe(token):
    word = tuple(token)
    pairs = get_pairs(word)
    if not pairs:
        return [token]
    while True:
        min_pair = min(pairs, key=lambda pair: bpe_ranks.get(pair, float('inf')))
        if min_pair not in bpe_ranks:
            break
        first, second = min_pair
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break
            if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = get_pairs(word)
    return word

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\S+|\s+(?!\S)|\s+""")

def encode(text):
    bpe_tokens = []
    for token in pat.findall(text):
        token = ''.join(b2u[b] for b in token.encode('utf-8'))
        bpe_tokens.extend(vocab[bpe_token] for bpe_token in bpe(token))
    return bpe_tokens

def decode(tokens):
    inv_vocab = {v: k for k, v in vocab.items()}
    text = ''.join(inv_vocab[t] for t in tokens)
    byte_array = bytearray([u2b[c] for c in text])
    return byte_array.decode('utf-8', errors='replace')

byte_to_token = {b: vocab[b2u[b]] for b in range(256)}
pair_to_merge = {}
for rank, (a, b) in enumerate(merges):
    id_a = vocab[a]
    id_b = vocab[b]
    id_merged = vocab[a + b]
    pair_to_merge[(id_a, id_b)] = (rank, id_merged)

def encode_fast(text):
    tokens = []
    for chunk in pat.findall(text):
        b = chunk.encode('utf-8')
        t_ids = [byte_to_token[byte] for byte in b]
        while len(t_ids) >= 2:
            best_rank = 1000000000
            best_idx = -1
            best_merged_id = -1
            for i in range(len(t_ids) - 1):
                p = (t_ids[i], t_ids[i+1])
                if p in pair_to_merge:
                    rank, mid = pair_to_merge[p]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
                        best_merged_id = mid
            if best_idx == -1:
                break
            t_ids = t_ids[:best_idx] + [best_merged_id] + t_ids[best_idx+2:]
        tokens.extend(t_ids)
    return tokens

def c_split(text):
    chunks = []
    i = 0
    n = len(text)
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
                    "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D"]
    while i < n:
        # 1. Contractions
        matched_contract = False
        for c in contractions:
            if text[i:i+len(c)] == c:
                chunks.append(c)
                i += len(c)
                matched_contract = True
                break
        if matched_contract:
            continue

        # 2. Optional leading space followed by word or non-space
        start = i
        has_lead_space = False
        if text[i] == ' ' and i + 1 < n and not text[i+1].isspace():
            has_lead_space = True
            i += 1

        if i < n and (text[i].isalnum() or text[i] == '_'):
            while i < n and (text[i].isalnum() or text[i] == '_'):
                i += 1
            chunks.append(text[start:i])
            continue
        elif has_lead_space and i < n and not text[i].isspace():
            # optional space + non-space
            while i < n and not text[i].isspace() and not (text[i].isalnum() or text[i] == '_'):
                # stop before contraction apostrophe
                if text[i] == '\'':
                    break
                i += 1
            chunks.append(text[start:i])
            continue
        else:
            i = start

        # 3. Punctuation / symbols without leading space
        if not text[i].isspace():
            start = i
            while i < n and not text[i].isspace() and not (text[i].isalnum() or text[i] == '_'):
                if text[i] == '\'':
                    break
                i += 1
            if i > start:
                chunks.append(text[start:i])
                continue

        # 4. Whitespace
        if text[i].isspace():
            start = i
            while i < n and text[i].isspace():
                i += 1
            chunks.append(text[start:i])
            continue
    return chunks


def encode_c_split(text):
    tokens = []
    for chunk in c_split(text):
        b = chunk.encode('utf-8')
        t_ids = [byte_to_token[byte] for byte in b]
        while len(t_ids) >= 2:
            best_rank = 1000000000
            best_idx = -1
            best_merged_id = -1
            for i in range(len(t_ids) - 1):
                p = (t_ids[i], t_ids[i+1])
                if p in pair_to_merge:
                    rank, mid = pair_to_merge[p]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
                        best_merged_id = mid
            if best_idx == -1:
                break
            t_ids = t_ids[:best_idx] + [best_merged_id] + t_ids[best_idx+2:]
        tokens.extend(t_ids)
    return tokens

if __name__ == '__main__':
    for test_str in [
        'Once upon a time',
        'Once upon a time, there was a little girl named Lily.',
        'Hello world! How are you doing today? 12345',
        'It\'s a sunny day! Isn\'t it?'
    ]:
        t1 = encode(test_str)
        t2 = encode_fast(test_str)
        t3 = encode_c_split(test_str)
        print(f'Test: {test_str}')
        print(f'  Match fast: {t1 == t2} | Match C-split: {t1 == t3} -> {t3}')



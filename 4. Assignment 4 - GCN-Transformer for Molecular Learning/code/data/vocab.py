import selfies as sf

import config


def build_vocab(train_records):
    symbols = set()
    for record in train_records:
        for symbol in sf.split_selfies(record["selfies"]):
            symbols.add(symbol)

    token_to_id = {token: i for i, token in enumerate(config.SPECIAL_TOKENS)}
    next_id = len(config.SPECIAL_TOKENS)
    for symbol in sorted(symbols):
        token_to_id[symbol] = next_id
        next_id += 1

    id_to_token = {i: token for token, i in token_to_id.items()}
    return token_to_id, id_to_token


def encode(selfies_string, token_to_id, add_specials=True):
    ids = [token_to_id.get(symbol, config.UNK_ID) for symbol in sf.split_selfies(selfies_string)]
    if add_specials:
        ids = [config.BOS_ID] + ids + [config.EOS_ID]
    return ids


def decode(ids, id_to_token, strip_specials=True):
    tokens = []
    for token_id in ids:
        token = id_to_token.get(int(token_id), config.UNK_TOKEN)
        if strip_specials and token in config.SPECIAL_TOKENS:
            if token == config.EOS_TOKEN:
                break
            continue
        tokens.append(token)
    return "".join(tokens)

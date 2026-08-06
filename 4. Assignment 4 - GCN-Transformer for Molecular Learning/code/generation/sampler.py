import config
from data.vocab import decode


def sample_molecules(transformer, id_to_token, n=None, temperature=1.0,
                     top_k=None, top_p=None, max_len=None, device=None):
    n = n or config.N_GENERATE
    token_id_sequences = transformer.generate(
        n=n, temperature=temperature, top_k=top_k, top_p=top_p,
        max_len=max_len, device=device)
    selfies_strings = [decode(ids, id_to_token, strip_specials=True)
                       for ids in token_id_sequences]
    return selfies_strings, token_id_sequences

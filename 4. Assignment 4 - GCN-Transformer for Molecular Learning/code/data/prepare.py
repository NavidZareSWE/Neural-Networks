import json

import config
from data.dataset import build_dataset
from data.vocab import build_vocab
import visualize


def divider(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def prepare_and_report(verbose=True, save=True):
    splits, stats = build_dataset()
    token_to_id, id_to_token = build_vocab(splits["train"])
    stats["vocab_size"] = len(token_to_id)

    if verbose:
        divider("SECTION 1 -- DATASET STATISTICS")
        print(f"Source CSV                    : {config.RAW_CSV}")
        print(f"Random seed                   : {config.SEED}")
        print(f"Original molecules            : {stats['n_original']}")
        print(f"Removed (failed RDKit parse)  : {stats['n_failed_parse']}")
        print(f"Removed (failed SELFIES enc.) : {stats['n_failed_selfies']}")
        print(f"Removed (SELFIES > {config.MAX_SELFIES_LEN} tokens): {stats['n_too_long']}")
        print(f"Removed (total)               : {stats['n_removed_total']}")
        print(f"Duplicated (canonical) removed: {stats['n_duplicates']}")
        print(f"Usable molecules              : {stats['n_usable']}")
        print(f"Split (train/val/test)        : "
              f"{stats['n_train']}/{stats['n_val']}/{stats['n_test']}")
        print(f"Vocabulary size (with specials): {stats['vocab_size']}")
        print(f"Average atom count            : {stats['avg_atoms']:.4f}")
        print(f"Average undirected bond count : {stats['avg_bonds']:.4f}")
        print(f"SELFIES length min/avg/max    : "
              f"{stats['sel_len_min']}/{stats['sel_len_avg']:.4f}/{stats['sel_len_max']}")

    if save:
        with open(config.DATASET_SPLITS_JSON, "w") as handle:
            json.dump(splits, handle)
        with open(config.VOCAB_JSON, "w") as handle:
            json.dump({"token_to_id": token_to_id}, handle, indent=2)
        with open(config.STATS_JSON, "w") as handle:
            json.dump(stats, handle, indent=2)
        lengths = [record["sel_len"] for part in splits.values() for record in part]
        visualize.plot_selfies_length_distribution(lengths)

    return splits, stats, token_to_id, id_to_token


if __name__ == "__main__":
    prepare_and_report()

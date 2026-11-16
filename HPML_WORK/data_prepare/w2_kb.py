"""W2 synthetic knowledge base: fictional entities, one document each, and a BM25 index over the documents.

python HPML_WORK/data/w2_kb.py      # writes out/w2/kb (built once, read-only afterwards)
"""

import argparse
import json
import random
from pathlib import Path

import bm25s
from common import WIKITEXT_REVISION, Lengths, fit_from_corpus, load_corpus_ids, rows_sha256
from w2_env import KnowledgeSearchEnv


KB_DIR = Path(__file__).resolve().parent / "out" / "w2" / "kb"
TYPES = ("person", "company", "city", "product")
# relation: (source type, target type, sentence)
RELATIONS = {
    "founder": ("company", "person", "{s} was founded by {t}."),
    "headquarters": ("company", "city", "{s} is headquartered in {t}."),
    "flagship_product": ("company", "product", "The flagship product of {s} is {t}."),
    "birth_city": ("person", "city", "{s} was born in {t}."),
    "employer": ("person", "company", "{s} works for {t}."),
    "mayor": ("city", "person", "The mayor of {s} is {t}."),
    "largest_employer": ("city", "company", "The largest employer in {s} is {t}."),
    "sister_city": ("city", "city", "The sister city of {s} is {t}."),
    "manufacturer": ("product", "company", "{s} is manufactured by {t}."),
    "designer": ("product", "person", "{s} was designed by {t}."),
}
# attribute: (entity type, low, high, sentence)
ATTRIBUTES = {
    "birth_year": ("person", 1900, 2005, "The birth year of {s} is {v}."),
    "height_cm": ("person", 150, 205, "{s} is {v} centimeters tall."),
    "founded_year": ("company", 1800, 2020, "{s} was established in {v}."),
    "employee_count": ("company", 20, 90000, "{s} has {v} employees."),
    "population": ("city", 2000, 900000, "{s} has a population of {v}."),
    "charter_year": ("city", 1100, 1900, "{s} received its city charter in {v}."),
    "release_year": ("product", 1990, 2025, "{s} was released in {v}."),
    "price_usd": ("product", 10, 5000, "{s} sells for {v} US dollars."),
}
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr", "st", "tr", "sh"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ei", "ou"]
CODAS = ["", "", "n", "r", "s", "l", "th"]


def make_word(rng, used_words):
    while True:
        word = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(2))
        if word not in used_words:
            used_words.add(word)
            return word.capitalize()


def make_names(rng, num_per_type, hard_negative_rate):
    """Two-word names, unique overall. A share of them reuse one word of an existing same-type name (hard negatives)."""
    used_words, names = set(), {}
    for entity_type in TYPES:
        names[entity_type] = []
        while len(names[entity_type]) < num_per_type:
            if names[entity_type] and rng.random() < hard_negative_rate:
                words = rng.choice(names[entity_type]).split()
                words[rng.randrange(2)] = make_word(rng, used_words)
                name = " ".join(words)
            else:
                name = f"{make_word(rng, used_words)} {make_word(rng, used_words)}"
            names[entity_type].append(name)
    return names


def build_entities(rng, lengths, corpus_ids, num_per_type, hard_negative_rate, L_doc):
    names = make_names(rng, num_per_type, hard_negative_rate)
    entities = []
    for entity_type in TYPES:
        for name in names[entity_type]:
            relations = {}
            for relation, (source, target, _) in RELATIONS.items():
                if source == entity_type:
                    relations[relation] = rng.choice([n for n in names[target] if n != name])
            attributes = {
                attribute: rng.randint(low, high)
                for attribute, (owner, low, high, _) in ATTRIBUTES.items()
                if owner == entity_type
            }
            entities.append({"name": name, "type": entity_type, "relations": relations, "attributes": attributes})
    for entity in entities:
        # Facts first (first half of the document), then unrelated text up to exactly L_doc tokens
        sentences = [RELATIONS[r][2].format(s=entity["name"], t=t) for r, t in entity["relations"].items()]
        sentences += [ATTRIBUTES[a][3].format(s=entity["name"], v=v) for a, v in entity["attributes"].items()]
        rng.shuffle(sentences)
        head = f"{entity['name']} is a {entity['type']}. " + " ".join(sentences) + "\nNotes:"
        entity["doc"] = fit_from_corpus(lengths, lengths.text, head, corpus_ids, rng, L_doc)
    return entities


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-per-type", type=int, default=1250)
    parser.add_argument("--hard-negative-rate", type=float, default=0.3)
    parser.add_argument("--L-doc", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--kb-dir", type=Path, default=KB_DIR)
    args = parser.parse_args()

    lengths = Lengths(args.tokenizer, [KnowledgeSearchEnv.search])
    corpus_ids = load_corpus_ids(lengths.tokenizer)
    rng = random.Random(f"w2-kb-{args.seed}")
    entities = build_entities(rng, lengths, corpus_ids, args.num_per_type, args.hard_negative_rate, args.L_doc)

    args.kb_dir.mkdir(parents=True, exist_ok=True)
    with open(args.kb_dir / "entities.jsonl", "w") as f:
        for entity in entities:
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([entity["doc"] for entity in entities], stopwords="en", show_progress=False))
    retriever.save(args.kb_dir / "bm25")
    meta = {
        "num_entities": len(entities),
        "num_per_type": args.num_per_type,
        "hard_negative_rate": args.hard_negative_rate,
        "L_doc": args.L_doc,
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "wikitext_revision": WIKITEXT_REVISION,
        "sha256": rows_sha256(entities),
    }
    (args.kb_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

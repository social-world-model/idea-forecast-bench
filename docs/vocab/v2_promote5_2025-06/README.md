# Locked concept vocabulary: v2 promote5, cutoff 2025-06

20 topics, built from papers dated 2024-10-01 .. 2025-05-31 (the training
side of cutoff 2025-06) with `config/vocab.yaml` (promote_min_count 5,
background_doc_frac 0.20, fine_threshold 0.90) on extraction store
`b493410c0021` (deepseek-v4-flash, prompt vocab_extract_v2.yaml, vectors
voyage-3-large). Reproduce with

    idea-forecast-bench vocab-export --topics <ids> --cutoff 2025-06 --output-dir <dir>

Files: one `<topic>.json` per topic (header fields + `concepts`) and
`vocabulary.csv` with every concept of every topic (21,832 rows).

| column | meaning |
|---|---|
| slot | object (what the work is done on), mechanism (how), problem (why) |
| label | the concept's name = its most frequent surface form |
| parent | the broader concept it belongs to; a folded concept's label is its parent |
| count | training papers containing the concept |
| doc_frac | count / training papers with a record |
| first_seen | earliest paper date in the training window |
| recent_count | papers in the 3 months before the cutoff |
| background | doc_frac >= 0.20: the topic itself, excluded from sampling |
| emerging | first seen in the last 3 months; kept even below 5 papers |
| variants | other surface forms merged into the concept |

Reading tips: the sampleable core is `background == False and count >= 5`;
`emerging == True` rows are the watch list of new phrasings (about half of
all rows, most with 1-2 papers). A topic's paper count sets its size:
reinforcement_learning has 3,719 concepts, protein_structure 283.

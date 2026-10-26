# GraphTango — educational Python implementation

This project is based on Ahmed, Siddique & Skadron, “GraphTango: A Hybrid Representation Format for Efficient Streaming Graph Updates and Analysis,” *International Journal of Parallel Programming* 52 (2024), 147–170. DOI: https://doi.org/10.1007/s10766-024-00768-x

## Included
- `graphtango_python.py`: hybrid vertex representation, edge insertion/deletion, BFS, PageRank, Dijkstra SSSP, connected components, SNAP download helper, and batch insertion benchmark.
- Dataset configuration for Orkut, LiveJournal, Wiki-topcats, and Wiki-talk.

## Paper's dataset inventory

| Paper label | SNAP dataset | Graph type | Vertices (paper) | Edges (paper) | URL |
|---|---|---|---:|---:|---|
| Orkut | `com-orkut.ungraph.txt.gz` | Undirected | 3.0 million | 117.2 million | https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz |
| LiveJournal | `soc-LiveJournal1.txt.gz` | Directed | 4.8 million | 69.0 million | https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz |
| Wiki | `wiki-topcats.txt.gz` | Directed | 1.8 million | 28.5 million | https://snap.stanford.edu/data/wiki-topcats.txt.gz |
| Talk | `wiki-Talk.txt.gz` | Directed | 2.4 million | 5.0 million | https://snap.stanford.edu/data/wiki-Talk.txt.gz |

The counts above are those reported in Table 2 of the paper, not counts recalculated by this script. Actual downloaded SNAP files may have different formatting or include comment/header lines; the parser skips comment lines.

## Run it

Requires Python 3.10+; standard library only.

```bash
python graphtango_python.py demo

# Download one dataset
python graphtango_python.py download --dataset wiki --data-dir data

# Smoke-test a small prefix (recommended before full data)
python graphtango_python.py benchmark --dataset wiki --data-dir data --limit 100000 --batch-size 10000

# Full input, processed in batches of one million edges
python graphtango_python.py benchmark --dataset livejournal --data-dir data --batch-size 1000000
```

The downloader stores files under `data/`. Large datasets may require substantial disk space and RAM. The benchmark currently reads the selected edge input into a Python list before shuffling it, so full-size datasets can require a lot of memory. Start with `--limit`.

## How the implementation maps to the paper

- Type 1: low-degree vertex, up to `TH0` neighbors stored in a small inline-style Python list.
- Type 2: medium-degree vertex, neighbors stored in a list and searched linearly.
- Type 3: high-degree vertex, neighbor list plus a destination-to-position dictionary for quick edge lookup.
- `TH0=7`, `TH1=32` are the paper's unweighted-graph settings for 64-bit vertex IDs; these are the defaults.
- Insertion checks for duplicate edges and updates an existing edge's weight.
- Deletion uses swap-with-last compaction, then remaps the vertex to the appropriate type.
- Analytics uses neighbor traversal, not the Type 3 lookup dictionary.

## Important scientific limitations

This is a **functional teaching prototype**, not a faithful performance reproduction of the paper's C++ system. Python cannot reproduce its cache-line-aligned `edgeMeta`, exact 64-bit memory layout, cache-line-aware open-addressing/double-hashing table, thread-local lock-free power-of-two memory pool, C++ memory allocation costs, or 12-core chunked parallel execution. In this prototype, Python lists and dictionaries stand in for the structures. Therefore, do not compare its throughput numbers directly with the paper's reported speedups.

The benchmark logs insertion time and vertex-type counts per batch. It does not yet implement the paper's full incremental analytics workload, delete-all phase, baselines (`AdListShared`, `AdListChunked`, Stinger, `DegAwareRHH`), or C++ hardware-counter profiling. For paper-faithful performance claims, use the authors' C++ implementation/SAGA-Bench and the paper's machine/setup where possible.

The paper's analytics are incremental and vertex-centric; this prototype's BFS, PageRank, SSSP, and connected-components functions are full recomputations for learning and correctness checks. Connected components on directed graphs defaults to weak connectivity. SSSP uses Dijkstra and therefore requires non-negative weights.

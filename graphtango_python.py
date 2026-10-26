#!/usr/bin/env python3
"""
GraphTango - single-file educational implementation + dataset runner.

Based on: Ahmed, Siddique & Skadron, "GraphTango: A Hybrid Representation Format for
Efficient Streaming Graph Updates and Analysis", Int. J. Parallel Programming 52 (2024).

One file does everything:
  * the GraphTango format (Type 1 / 2 / 3 vertices, promote/demote, doubling/halving)
  * the paper-style cache-friendly hash table (optional index for Type 3)
  * two baselines (array-only, hash-only) for comparison
  * BFS / PageRank / SSSP / connected components
  * SNAP dataset download + parsing, and an insert -> analytics -> delete benchmark

Quick start (Python 3.10+, standard library only):
  python graphtango_python.py demo
  python graphtango_python.py trace
  python graphtango_python.py run --dataset talk                 # download + full experiment
  python graphtango_python.py run --dataset all --limit 200000   # all four datasets
  python graphtango_python.py benchmark --dataset wiki --limit 100000 --batch-size 10000
  python graphtango_python.py test

NOT a performance reproduction of the C++ paper: Python has no cache-line layout,
lock-free allocator or 12-core execution. Compare the machine-independent "work"
column (edge comparisons + hash probes per update), not wall-clock speed.
"""
from __future__ import annotations  # make type hints lazy so annotations work on older Python 3.x

import argparse
import gzip
import heapq
import math
import random
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

Edge = Tuple[int, float]  # an edge = (neighbor vertex id, weight)

# --------------------------------------------------------------------------- datasets
DATASETS = {  # registry of SNAP datasets: download URL, directedness, paper-reported size, description
    "orkut": {"url": "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz",
              "directed": False, "expected_vertices_m": 3.0, "expected_edges_m": 117.2,
              "description": "Orkut social network (undirected, short-tailed)"},
    "livejournal": {"url": "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz",
                    "directed": True, "expected_vertices_m": 4.8, "expected_edges_m": 69.0,
                    "description": "LiveJournal social network (short-tailed)"},
    "wiki": {"url": "https://snap.stanford.edu/data/wiki-topcats.txt.gz",
             "directed": True, "expected_vertices_m": 1.8, "expected_edges_m": 28.5,
             "description": "Wikipedia hyperlinks, Wiki-topcats (heavy-tailed)"},
    "talk": {"url": "https://snap.stanford.edu/data/wiki-Talk.txt.gz",
             "directed": True, "expected_vertices_m": 2.4, "expected_edges_m": 5.0,
             "description": "Wikipedia discussion network, Wiki-talk (heavy-tailed)"},
}


def parse_edges(path: Path, limit: Optional[int] = None) -> Iterator[Tuple[int, int, float]]:  # stream edges lazily from a SNAP file so huge graphs never sit fully in memory
    """Read a SNAP edge list (.txt or .gz); skips '#'/'%' comments and malformed lines."""
    opener = gzip.open if str(path).endswith(".gz") else open  # pick gzip reader for .gz files, plain open() otherwise
    count = 0  # 'rt' = read as text; errors='replace' tolerates odd bytes
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:  # iterate the file line by line
        for line in f:
            line = line.strip()
            if not line or line[0] in "#%":  # skip blank lines and '#'/'%' comment lines
                continue
            parts = line.split()  # SNAP files are whitespace-separated: src dst [weight]
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
                w = float(parts[2]) if len(parts) >= 3 else 1.0  # weight defaults to 1.0 for unweighted datasets
            except ValueError:
                continue  # skip malformed lines (e.g. non-numeric tokens)
            yield u, v, w  # hand one edge to the caller
            count += 1
            if limit is not None and count >= limit:  # stop early once --limit edges have been read
                return


def sample_edges(path: Path, limit: int, seed: int = 42) -> List[Tuple[int, int, float]]:  # uniform random sample of `limit` edges
    """Reservoir-sample `limit` edges from the whole file in one pass.

    SNAP files are sorted by source vertex, so the first `limit` lines contain only a few thousand
    hub-like sources and distort the Type1/2/3 mix. The paper shuffles the entire dataset, so a
    uniform sample is much closer to its setup.
    """
    rng = random.Random(seed)  # seeded -> reproducible sample
    res: List[Tuple[int, int, float]] = []  # the reservoir
    for i, e in enumerate(parse_edges(path)):  # i = number of edges seen before this one
        if i < limit:  # fill the reservoir first
            res.append(e)
        else:
            j = rng.randrange(i + 1)  # keep edge i with probability limit/(i+1)
            if j < limit:
                res[j] = e  # replace a random earlier pick
    return res


def download_dataset(name: str, data_dir: Path) -> Path:  # download a dataset once and cache it in data_dir
    spec = DATASETS[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / spec["url"].rsplit("/", 1)[-1]  # local filename = last part of the URL
    if target.exists():  # already cached -> skip the download
        print(f"Already downloaded: {target}")
        return target
    print(f"Downloading {name}: {spec['url']}")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "GraphTango-educational/2.0"})
    tmp = target.with_name(target.name + ".part")  # download to a .part file so an interrupted download never looks complete
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)  # read the stream in 1 MiB chunks
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 2**20:.1f} / {total / 2**20:.1f} MiB", end="", flush=True)  # live progress indicator on one line
        print()
    except Exception as e:                                   # network problems are common  # network errors are common; fail with manual-download instructions
        tmp.unlink(missing_ok=True)
        sys.exit(f"Download failed ({e}).\nDownload the file manually from\n  {spec['url']}\n"
                 f"and place it in {data_dir}/ - then re-run.")
    tmp.rename(target)  # atomically move the finished file into place
    print(f"Saved: {target} ({target.stat().st_size / 2**20:.1f} MiB)")
    return target


def synthetic_edges(kind: str, vertices: int, edges: int, seed: int = 1) -> List[Tuple[int, int, float]]:  # generate offline test edges (no download needed)
    """Offline test data. 'short' = uniform endpoints, 'heavy' = a few hub vertices."""
    rng = random.Random(seed)  # seeded RNG -> reproducible graphs
    out = []
    for _ in range(edges):
        u = int(vertices * rng.random() ** 4) if kind == "heavy" else rng.randrange(vertices)  # 'heavy': rng**4 skews sources toward small ids => a few hub vertices; 'short': uniform
        v = rng.randrange(vertices)
        if u != v:  # drop self-loops
            out.append((u, v, 1.0))
    return out


def batch_max_degree(edges: list, batch_size: int) -> int:  # measure skew: highest per-vertex update count within one batch
    """Most updates any one source vertex gets inside a single batch (the paper's tail measure)."""
    best = 0
    for s in range(0, len(edges), batch_size):
        counts: Dict[int, int] = {}  # count updates per source vertex in this batch
        for u, _, _ in edges[s:s + batch_size]:
            counts[u] = counts.get(u, 0) + 1
        if counts:
            best = max(best, max(counts.values()))
    return best


# ----------------------------------------------------------------------- hash indexes
_MASK = (1 << 64) - 1  # 64-bit mask used by the integer hash below
_EMPTY, _TOMB = object(), object()  # unique sentinels: never-used slot vs. deleted slot (tombstone)


def _pow2(n: int) -> int:  # round n up to the next power of two
    p = 1
    while p < n:
        p <<= 1
    return p


def _mix(x: int, salt: int) -> int:  # SplitMix64-style integer mixer; 'salt' yields independent hash functions
    z = (x + salt * 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


class DictIndex:  # simple index backed by a Python dict: dst -> position in the edge array
    """dst -> array position using a Python dict (fast default)."""

    def __init__(self, min_slots: int = 16) -> None:
        self._d: Dict[int, int] = {}
        self.ops = self.probes = 0  # ops/probes counters feed the machine-independent 'work' metric

    def get(self, key: int) -> Optional[int]:
        self.ops += 1; self.probes += 1
        return self._d.get(key)

    def put(self, key: int, value: int) -> None:
        self.ops += 1; self.probes += 1
        self._d[key] = value

    def remove(self, key: int) -> bool:
        self.ops += 1; self.probes += 1
        return self._d.pop(key, None) is not None

    def __len__(self) -> int:
        return len(self._d)


class CacheFriendlyHashTable:  # paper-style table: probes stay inside one 8-slot group (a cache line) as long as possible
    """Paper Section 5: h(key,i) = N*h1(key, i//N) + h2(key, i%N), N = 8 slots (one cache line).

    h1 = double hashing over groups, h2 = linear probing inside a group, load factor <= 0.5.
    `lines` counts the distinct 8-slot groups (cache lines) touched.
    """

    def __init__(self, min_slots: int = 16, group: int = 8, max_load: float = 0.5) -> None:
        self.group, self.max_load = group, max_load
        self.min_slots = max(_pow2(min_slots), group * 2)  # ensure at least two groups
        self.ops = self.probes = self.lines = 0
        self._alloc(self.min_slots)  # allocate empty storage

    def _alloc(self, slots: int) -> None:  # (re)create the key/value slot arrays
        self.slots, self.groups = slots, slots // self.group
        self._keys, self._vals = [_EMPTY] * slots, [0] * slots  # keys start EMPTY, values 0
        self.live = self.used = 0  # live = real keys, used = live + tombstones

    def _sequence(self, key: int):  # generate the probe order for a key: double hashing between groups, linear within a group
        g, n = self.groups, self.group
        a, b, c = _mix(key, 1), _mix(key, 2) | 1, _mix(key, 3)  # a, b, c = three independent hashes (b forced odd so it is coprime with the power-of-two group count)
        for i in range(self.slots):
            yield ((a + (i // n) * b) % g) * n + (c + i % n) % n, i  # slot = group chosen by double hashing * group size + linear offset within the group

    def _locate(self, key: int) -> Tuple[int, int]:  # find a key: returns (slot_of_key, -1) if found, else (-1, slot_to_insert_into)
        first_tomb, probes = -1, 0
        for slot, i in self._sequence(key):
            probes = i + 1
            k = self._keys[slot]
            if k is _EMPTY:  # EMPTY slot ends the probe chain => key absent
                self._acct(probes)
                return -1, (first_tomb if first_tomb >= 0 else slot)
            if k is _TOMB:  # remember the first tombstone so inserts can reuse it
                if first_tomb < 0:
                    first_tomb = slot
            elif k == key:  # found the key
                self._acct(probes)
                return slot, -1
        self._acct(probes)  # table fully probed without finding an empty slot
        return -1, first_tomb

    def _acct(self, probes: int) -> None:  # record statistics: ops, probes, and distinct cache lines touched
        self.ops += 1; self.probes += probes; self.lines += (probes - 1) // self.group + 1

    def _rebuild(self, slots: int) -> None:  # rehash all live entries into a table with the given slot count (grow, shrink or clean tombstones)
        items = [(k, v) for k, v in zip(self._keys, self._vals) if k is not _EMPTY and k is not _TOMB]  # collect live items, skipping empties and tombstones
        self._alloc(max(slots, self.min_slots))  # fresh storage, never smaller than the minimum
        for k, v in items:
            for slot, _ in self._sequence(k):
                if self._keys[slot] is _EMPTY:
                    self._keys[slot], self._vals[slot] = k, v
                    self.live += 1; self.used += 1
                    break

    def get(self, key: int) -> Optional[int]:  # lookup: value for key or None
        slot, _ = self._locate(key)
        return None if slot < 0 else self._vals[slot]

    def put(self, key: int, value: int) -> None:  # insert or overwrite
        slot, free = self._locate(key)
        if slot >= 0:  # key exists -> overwrite its value
            self._vals[slot] = value
            return
        if self._keys[free] is _EMPTY:  # consuming a never-used slot increases 'used'; reusing a tombstone does not
            self.used += 1
        self._keys[free], self._vals[free] = key, value
        self.live += 1
        if self.used > self.max_load * self.slots:  # load factor exceeded (counting tombstones) -> rebuild
            grow = self.live > self.max_load * self.slots / 2  # grow only if many keys are really alive; otherwise just purge tombstones at same size
            self._rebuild(self.slots * 2 if grow else self.slots)

    def remove(self, key: int) -> bool:  # delete a key; True if it was present
        slot, _ = self._locate(key)
        if slot < 0:
            return False
        self._keys[slot] = _TOMB  # leave a tombstone so later probe chains are not broken
        self.live -= 1
        if self.slots > self.min_slots and self.live < self.slots * self.max_load / 4:  # shrink when the table becomes sparse (live < 1/8 of slots)
            self._rebuild(self.slots // 2)
        return True

    def __len__(self) -> int:
        return self.live


INDEX_IMPLS = {"dict": DictIndex, "cfh": CacheFriendlyHashTable}  # name -> index class, selectable via --index


# ------------------------------------------------------------------------ graph base
class _Graph:  # base class: shared add/remove API and analytics; subclasses supply storage
    """Shared update API + analytics (they only use neighbors(), never the hash index)."""
    name = "graph"

    def __init__(self, directed: bool = True) -> None:
        self.directed = directed
        self.edge_count = 0

    def add_edge(self, src: int, dst: int, weight: float = 1.0) -> bool:  # add an edge; return True if it is new
        """Insert edge. True if new; False if it existed (its weight is updated)."""
        new = self._insert_one(src, dst, weight)  # store src -> dst
        if not self.directed and src != dst:  # undirected graphs also store the reverse direction
            self._insert_one(dst, src, weight)
        if new:  # edge_count counts logical edges (undirected edge counted once)
            self.edge_count += 1
        return new

    def remove_edge(self, src: int, dst: int) -> bool:  # delete an edge; return True if it existed
        removed = self._remove_one(src, dst)
        if removed:
            self.edge_count -= 1
            if not self.directed and src != dst:
                self._remove_one(dst, src)
        return removed

    @property  # number of vertices that own an adjacency structure
    def num_vertices(self) -> int:
        return sum(1 for _ in self.vertex_ids())

    # ---- analytics (full recomputation; the paper uses incremental versions)
    def bfs(self, source: int) -> Dict[int, int]:  # breadth-first search: hop distance from source to every reachable vertex
        dist = {source: 0}  # source is at distance 0
        q = deque([source])  # FIFO queue of vertices to expand
        while q:
            u = q.popleft()
            for v, _ in self.neighbors(u):
                if v not in dist:  # first time a vertex is seen = shortest hop distance
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def pagerank(self, iterations: int = 20, damping: float = 0.85, tolerance: float = 1e-8) -> Dict[int, float]:  # power-iteration PageRank with dangling-node handling; full recomputation
        nodes = set(self.vertex_ids())  # collect all vertices, including ones that are only edge targets
        for u in list(nodes):
            nodes.update(v for v, _ in self.neighbors(u))
        if not nodes:
            return {}
        n = len(nodes)
        rank = {u: 1.0 / n for u in nodes}  # start with a uniform rank
        outdeg = {u: self.degree(u) for u in nodes}  # out-degree of each vertex
        for _ in range(iterations):
            dangling = sum(rank[u] for u in nodes if outdeg[u] == 0)  # rank mass sitting on sink vertices is spread uniformly
            base = (1 - damping) / n + damping * dangling / n  # teleport term + redistributed dangling mass
            new = {u: base for u in nodes}
            for u in nodes:
                if outdeg[u]:
                    share = damping * rank[u] / outdeg[u]  # each out-neighbor receives an equal share
                    for v, _ in self.neighbors(u):
                        new[v] += share
            delta = sum(abs(new[u] - rank[u]) for u in nodes)  # L1 change in ranks; used as convergence test
            rank = new
            if delta < tolerance:  # stop early when converged
                break
        return rank

    def sssp(self, source: int) -> Dict[int, float]:  # single-source shortest paths with Dijkstra's algorithm
        """Dijkstra; weights must be non-negative."""
        dist = {source: 0.0}  # best known distance per vertex
        heap = [(0.0, source)]  # min-heap of (distance, vertex)
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist.get(u):  # stale heap entry (a shorter path was already found) -> skip
                continue
            for v, w in self.neighbors(u):
                if w < 0:
                    raise ValueError("SSSP/Dijkstra requires non-negative weights")  # Dijkstra is invalid for negative weights
                nd = d + w
                if nd < dist.get(v, math.inf):  # relax edge u->v if this path is shorter
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def connected_components(self, weak_for_directed: bool = True) -> Dict[int, int]:  # label connected components (weakly connected for directed graphs)
        adj: Dict[int, Set[int]] = {}  # build an adjacency-set view, symmetric when treating direction as ignorable
        for u in self.vertex_ids():
            adj.setdefault(u, set())
            for v, _ in self.neighbors(u):
                adj[u].add(v)
                adj.setdefault(v, set())
                if not self.directed or weak_for_directed:
                    adj[v].add(u)
        labels: Dict[int, int] = {}  # component label per vertex
        cid = 0
        for start in adj:
            if start in labels:
                continue
            cid += 1  # new component id for each unvisited start vertex
            labels[start] = cid
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in labels:
                        labels[v] = cid
                        q.append(v)
        return labels


# ------------------------------------------------------------------------ GraphTango
class Vertex:  # per-vertex record; __slots__ avoids a per-object __dict__ to save memory
    __slots__ = ("kind", "edges", "index", "capacity")  # kind = Type 1/2/3, edges = edge array, index = optional hash index, capacity = modelled array size

    def __init__(self, capacity: int) -> None:
        self.kind, self.edges, self.index, self.capacity = 1, [], None, capacity  # new vertices start as Type 1


class GraphTango(_Graph):  # the hybrid format from the paper: representation chosen by vertex degree
    """Hybrid format. Type 1: degree <= TH0 (edges 'inline'); Type 2: <= TH1 (array);
    Type 3: > TH1 (array + hash index). Arrays double when full and halve at 1/4 full;
    deletion swaps the last entry into the hole.

    hysteresis=False (default) demotes exactly at the thresholds. hysteresis=True keeps a
    Type 3 vertex until degree <= TH1/2 (avoids rebuilding the index near TH1).
    """
    name = "GraphTango"

    def __init__(self, directed: bool = True, th0: int = 7, th1: int = 32,  # th0/th1 = degree thresholds between Type 1/2 and Type 2/3
                 index: str = "dict", hysteresis: bool = False,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        super().__init__(directed)
        if th0 < 1 or th1 <= th0:
            raise ValueError("Require 1 <= th0 < th1")  # thresholds must be ordered
        if index not in INDEX_IMPLS:
            raise ValueError(f"index must be one of {sorted(INDEX_IMPLS)}")
        self.th0, self.th1, self.hysteresis = th0, th1, hysteresis
        self._index_cls = INDEX_IMPLS[index]
        self._min_cap = _pow2(th0 + 1)  # smallest array capacity that holds more than th0 edges
        self.on_event = on_event  # optional callback for logging promotions/resizes (used by trace())
        self.vertices: Dict[int, Vertex] = {}  # vertex id -> Vertex record
        self.lookups = self.comparisons = 0  # counters for lookups and linear-scan comparisons
        self.promotions = self.demotions = self.array_resizes = 0  # counters for structural changes
        self._retired = [0, 0]                       # ops, probes of dropped indexes  # ops/probes of dropped indexes, kept so total work stays accurate

    # -- internals
    def _emit(self, msg: str) -> None:  # forward a human-readable event to the callback, if any
        if self.on_event:
            self.on_event(msg)

    def _find(self, v: Vertex, dst: int) -> int:  # return position of dst in the vertex's edge array, or -1
        self.lookups += 1  # count one lookup
        if v.kind == 3:  # Type 3: O(1) via hash index
            pos = v.index.get(dst)
            return -1 if pos is None else pos
        e = v.edges  # Types 1/2: linear scan of the array
        for i in range(len(e)):
            if e[i][0] == dst:
                self.comparisons += i + 1  # count comparisons made (machine-independent cost)
                return i
        self.comparisons += len(e)
        return -1

    def _build_index(self, v: Vertex) -> None:  # build a hash index over the current edge array
        self._drop_index(v)  # bank the old index's work counters before it is replaced (else work() undercounts)
        idx = self._index_cls(min_slots=2 * v.capacity)  # size the table at 2x the array capacity
        for i, (d, _) in enumerate(v.edges):
            idx.put(d, i)  # map each neighbor id -> its array position
        v.index = idx

    def _drop_index(self, v: Vertex) -> None:  # discard the hash index (vertex no longer Type 3)
        if v.index is not None:
            self._retired[0] += v.index.ops  # bank its ops/probes before dropping it
            self._retired[1] += v.index.probes
            v.index = None

    def _target_kind(self, v: Vertex) -> int:  # which Type the vertex should have given its current degree
        d = len(v.edges)
        if d <= self.th0:
            return 1  # low degree -> Type 1
        if v.kind == 3 and self.hysteresis:  # with hysteresis a Type 3 vertex stays Type 3 until degree <= th1/2
            return 2 if d <= self.th1 // 2 else 3
        return 2 if d <= self.th1 else 3

    def _retype(self, u: int, v: Vertex) -> None:  # promote or demote the vertex if its degree crossed a threshold
        new, old, d = self._target_kind(v), v.kind, len(v.edges)
        if new == old:
            return  # no change needed
        if new > old:
            self.promotions += 1  # count promotions vs demotions
        else:
            self.demotions += 1
        if new == 1:  # demoting to Type 1: no index, fixed small capacity
            self._drop_index(v)
            v.capacity = self.th0
        else:
            if old == 1:  # leaving Type 1: allocate a power-of-two array
                v.capacity = max(_pow2(d), self._min_cap)
            if new == 3:  # entering Type 3: build the hash index
                self._build_index(v)
            else:
                self._drop_index(v)
        v.kind = new
        self._emit(f"vertex {u}: {'promoted' if new > old else 'demoted'} Type {old} -> Type {new} (degree {d})")

    # -- updates
    def _insert_one(self, src: int, dst: int, weight: float) -> bool:  # insert (src -> dst); returns True if new
        v = self.vertices.get(src)
        if v is None:  # create the vertex lazily on first edge
            v = self.vertices[src] = Vertex(self.th0)
        pos = self._find(v, dst)  # check for an existing edge first
        if pos >= 0:  # duplicate: just update the weight
            v.edges[pos] = (dst, weight)
            return False
        v.edges.append((dst, weight))  # append to the end of the edge array
        if v.kind == 3:  # keep the hash index in sync
            v.index.put(dst, len(v.edges) - 1)
        if v.kind >= 2 and len(v.edges) > v.capacity:  # array full => double capacity (amortized O(1) appends)
            while len(v.edges) > v.capacity:
                v.capacity *= 2
            self.array_resizes += 1
            self._emit(f"vertex {src}: array full -> capacity doubled to {v.capacity}")
            if v.kind == 3:
                self._build_index(v)
        self._retype(src, v)  # check whether the vertex must change Type
        return True

    def _remove_one(self, src: int, dst: int) -> bool:  # remove (src -> dst); returns True if it existed
        v = self.vertices.get(src)
        if v is None:
            return False
        pos = self._find(v, dst)
        if pos < 0:
            return False
        last = v.edges[-1]  # last edge fills the hole left by the removed one
        v.edges[pos] = last                    # swap-with-last compaction  # swap-with-last compaction keeps the array dense, O(1)
        v.edges.pop()
        if v.kind == 3:  # update the hash index for Type 3
            v.index.remove(dst)
            if pos < len(v.edges):  # the moved edge now lives at position 'pos'
                v.index.put(last[0], pos)
        if v.kind >= 2 and v.capacity > self._min_cap and len(v.edges) * 4 <= v.capacity:  # shrink the array when it is only 1/4 full
            v.capacity //= 2
            self.array_resizes += 1
            self._emit(f"vertex {src}: array 1/4 full -> capacity halved to {v.capacity}")
            if v.kind == 3:
                self._build_index(v)
        self._retype(src, v)
        return True

    # -- queries
    def neighbors(self, u: int) -> Iterator[Edge]:  # iterate a vertex's out-edges straight from the array (never touches the index)
        v = self.vertices.get(u)
        if v is not None:
            yield from v.edges                 # array only - the hash index is never read

    def degree(self, u: int) -> int:  # out-degree
        v = self.vertices.get(u)
        return len(v.edges) if v else 0

    def vertex_type(self, u: int) -> int:  # current Type (1, 2 or 3) of a vertex
        v = self.vertices.get(u)
        return v.kind if v else 1

    def vertex_ids(self):  # iterator over stored vertex ids
        return iter(self.vertices)

    def type_counts(self) -> List[int]:  # how many vertices are Type 1, 2 and 3
        c = [0, 0, 0]
        for v in self.vertices.values():
            c[v.kind - 1] += 1
        return c

    def describe(self, u: int) -> str:  # human-readable description of a vertex's representation (used by trace())
        v = self.vertices.get(u)
        if v is None:
            return f"vertex {u}: (not stored)"
        shown = ", ".join(str(d) for d, _ in v.edges[:12]) + (" ..." if len(v.edges) > 12 else "")  # show at most the first 12 neighbors
        if v.kind == 1:
            return f"vertex {u}: Type 1 | degree={len(v.edges)} | inline [{shown}]"
        extra = f" + hash index ({len(v.index)} keys)" if v.kind == 3 else ""
        return f"vertex {u}: Type {v.kind} | degree={len(v.edges)} | array cap={v.capacity} [{shown}]{extra}"

    def work(self) -> int:  # total work = linear-scan comparisons + hash probes (machine-independent cost)
        ops_probes = self._retired[1] + sum(v.index.probes for v in self.vertices.values() if v.index)
        return self.comparisons + ops_probes

    def estimated_bytes(self, id_bytes: int = 8) -> int:  # model memory with the paper's layout (not measured Python memory)
        """Modelled (not measured) memory using the paper's layout."""
        total = 0
        for v in self.vertices.values():
            total += 64  # assume 64 bytes of per-vertex metadata/inline storage
            if v.kind >= 2:
                total += v.capacity * id_bytes  # Types 2/3: edge array
            if v.kind == 3:
                total += 2 * v.capacity * id_bytes  # Type 3: hash index with ~2x slots, key + value
        return total


# --------------------------------------------------------------------------- baselines
class AdListGraph(_Graph):  # baseline 1: plain adjacency list scanned linearly
    """Array per vertex, always scanned linearly (idea of AdListShared/Chunked)."""
    name = "AdjList"

    def __init__(self, directed: bool = True) -> None:
        super().__init__(directed)
        self.adj: Dict[int, List[Edge]] = {}  # vertex -> list of (dst, weight)
        self.comparisons = 0

    def _insert_one(self, src, dst, weight):  # insert; scans the whole list to detect duplicates
        a = self.adj.setdefault(src, [])
        for i in range(len(a)):
            if a[i][0] == dst:
                self.comparisons += i + 1
                a[i] = (dst, weight)
                return False
        self.comparisons += len(a)
        a.append((dst, weight))  # append new edge
        return True

    def _remove_one(self, src, dst):  # delete by scanning, then swap-with-last
        a = self.adj.get(src)
        if not a:
            return False
        for i in range(len(a)):
            if a[i][0] == dst:
                self.comparisons += i + 1
                a[i] = a[-1]
                a.pop()
                return True
        self.comparisons += len(a)
        return False

    def neighbors(self, u):  # yield out-edges
        yield from self.adj.get(u, ())

    def degree(self, u):
        return len(self.adj.get(u, ()))

    def vertex_ids(self):
        return iter(self.adj)

    def work(self):
        return self.comparisons

    def estimated_bytes(self, id_bytes: int = 8):
        return sum(24 + _pow2(max(len(a), 1)) * id_bytes for a in self.adj.values())


class HashGraph(_Graph):  # baseline 2: every vertex has its own hash table (a Python dict here)
    """Hash table per vertex, always (idea of DegAwareRHH)."""
    name = "HashOnly"

    def __init__(self, directed: bool = True) -> None:
        super().__init__(directed)
        self.adj: Dict[int, Dict[int, float]] = {}  # vertex -> {dst: weight}
        self.probes = 0

    def _insert_one(self, src, dst, weight):
        d = self.adj.setdefault(src, {})
        self.probes += 1
        new = dst not in d  # O(1) membership test
        d[dst] = weight
        return new

    def _remove_one(self, src, dst):
        d = self.adj.get(src)
        self.probes += 1
        if d is None or dst not in d:
            return False
        del d[dst]
        return True

    def neighbors(self, u):
        yield from self.adj.get(u, {}).items()

    def degree(self, u):
        return len(self.adj.get(u, ()))

    def vertex_ids(self):
        return iter(self.adj)

    def work(self):
        return self.probes

    def estimated_bytes(self, id_bytes: int = 8):
        return sum(24 + _pow2(max(2 * len(d), 2)) * 2 * id_bytes for d in self.adj.values())


# -------------------------------------------------------------------------- experiment
def load_graph(path: Path, directed: bool, th0: int, th1: int, limit: Optional[int] = None) -> GraphTango:  # build a GraphTango from a dataset file
    g = GraphTango(directed=directed, th0=th0, th1=th1)
    for u, v, w in parse_edges(path, limit=limit):
        g.add_edge(u, v, w)
    return g


def run_experiment(label: str, edges: list, directed: bool, batch_size: int, th0: int, th1: int,  # run the insert -> analytics -> delete benchmark on all three formats
                   index: str, seed: int, csv_path: Optional[str] = None, plot_path: Optional[str] = None) -> List[dict]:
    """Insert in batches -> BFS/PageRank -> delete in batches, for all three formats."""
    rng = random.Random(seed)
    edges = list(edges)  # shuffle edges so arrival order is random (seeded)
    rng.shuffle(edges)
    n = len(edges)
    if batch_size > n:
        print(f"(note: batch_size {batch_size:,} > {n:,} edges -> only one batch; use a smaller --batch-size)")
    print(f"\n=== {label}: edges={n:,} batch_size={batch_size:,} "
          f"max updates to one vertex in a batch={batch_since_max(edges, batch_size):,} ===")
    makers = {"GraphTango": lambda: GraphTango(directed, th0, th1, index=index),  # factories for the three formats under comparison
              "AdjList": lambda: AdListGraph(directed), "HashOnly": lambda: HashGraph(directed)}
    rows = []
    for name, make in makers.items():
        g = make()
        verbose = name == "GraphTango"  # only GraphTango prints per-batch CSV rows
        if verbose:
            print("batch,edges_in_batch,insert_seconds,edges_per_second,vertices,Type1,Type2,Type3")
        ins_t, w0 = 0.0, g.work()
        for b, s in enumerate(range(0, n, batch_size), 1):  # insert in batches of batch_size
            batch = edges[s:s + batch_size]
            t0 = time.perf_counter()
            for u, v, w in batch:
                g.add_edge(u, v, w)
            dt = max(time.perf_counter() - t0, 1e-12)  # perf_counter is a high-resolution timer
            ins_t += dt
            if verbose:
                c = g.type_counts()
                print(f"{b},{len(batch)},{dt:.4f},{len(batch) / dt:.0f},{g.num_vertices},{c[0]},{c[1]},{c[2]}")
        ins_work = g.work() - w0  # work done during insertion only
        mem, stored = g.estimated_bytes(), g.edge_count
        hub = max(g.vertex_ids(), key=g.degree)  # highest-degree vertex = BFS source
        t0 = time.perf_counter(); reach = len(g.bfs(hub)); bfs_t = time.perf_counter() - t0
        t0 = time.perf_counter(); g.pagerank(iterations=3); pr_t = time.perf_counter() - t0
        c = g.type_counts() if isinstance(g, GraphTango) else None  # vertex type counts (GraphTango only)
        order = list(edges)
        rng.shuffle(order)  # delete every edge in a fresh random order
        del_t, w1 = 0.0, g.work()
        for s in range(0, n, batch_size):
            t0 = time.perf_counter()
            for u, v, _ in order[s:s + batch_size]:
                g.remove_edge(u, v)
            del_t += time.perf_counter() - t0
        rows.append({"dataset": label, "format": name, "edges_stored": stored,  # collect one result row per format
                     "insert_Meps": n / ins_t / 1e6, "delete_Meps": n / max(del_t, 1e-12) / 1e6,
                     "insert_work_per_op": ins_work / n, "delete_work_per_op": (g.work() - w1) / n,
                     "bfs_ms": bfs_t * 1e3, "pagerank3_ms": pr_t * 1e3, "bfs_reach": reach,
                     "bytes_per_edge": mem / max(stored, 1), "edges_left": g.edge_count,
                     "type1": c[0] if c else "", "type2": c[1] if c else "", "type3": c[2] if c else ""})
    print_table(rows)  # print the comparison table
    if csv_path:  # optionally append results to a CSV file
        import csv
        new_file = not Path(csv_path).exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new_file:
                w.writeheader()
            w.writerows(rows)
        print(f"Appended results to {csv_path}")
    if plot_path:  # optionally draw bar charts
        make_plot(rows, plot_path, label)
    return rows


def batch_since_max(edges, batch_size):          # small alias keeps the f-string above short  # short alias for batch_max_degree
    return batch_max_degree(edges, batch_size)


def print_table(rows: List[dict]) -> None:  # pretty-print the comparison table
    cols = [("format", "Format", "{}"), ("insert_Meps", "Ins M e/s", "{:.3f}"), ("delete_Meps", "Del M e/s", "{:.3f}"),  # (key, header, format) for each column
            ("insert_work_per_op", "Ins work/op", "{:.1f}"), ("delete_work_per_op", "Del work/op", "{:.1f}"),
            ("bfs_ms", "BFS ms", "{:.1f}"), ("pagerank3_ms", "PR(3) ms", "{:.1f}"), ("bytes_per_edge", "B/edge*", "{:.1f}")]
    widths = [max(len(h), *(len(f.format(r[k])) for r in rows)) for k, h, f in cols]  # column width = widest of header and values
    line = "  ".join(h.ljust(w) for (_, h, _), w in zip(cols, widths))
    print("\n" + line + "\n" + "-" * len(line))
    for r in rows:
        print("  ".join(f.format(r[k]).ljust(w) for (k, _, f), w in zip(cols, widths)))
    g = rows[0]
    print(f"\n* B/edge is a MODEL of the paper's memory layout, not measured Python memory.")
    print("  work/op = edge comparisons + hash probes per update (lower is better; use this, not wall time).")
    print(f"  GraphTango vertex types: Type1={g['type1']:,} Type2={g['type2']:,} Type3={g['type3']:,}")


def make_plot(rows: List[dict], path: str, title: str) -> None:  # bar charts of the key metrics (needs matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend: save to file without a window
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping plot; pip install matplotlib)")
        return
    metrics = [("insert_work_per_op", "Insert work / op"), ("delete_work_per_op", "Delete work / op"),
               ("bfs_ms", "BFS time (ms)"), ("bytes_per_edge", "Modelled bytes / edge")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, (k, lab) in zip(axes, metrics):
        ax.bar([r["format"] for r in rows], [r[k] for r in rows], color=["black", "grey", "lightgrey"], edgecolor="black")
        ax.set_title(lab, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Wrote {path}")


# ------------------------------------------------------------------------------ demos
def demo() -> None:  # small walkthrough: tiny graph, thresholds 2 and 4, each analytic once
    g = GraphTango(directed=True, th0=2, th1=4)
    for e in [(1, 2, 1), (1, 3, 2), (1, 4, 1), (1, 5, 3), (2, 3, 1), (3, 4, 1), (4, 5, 1)]:
        g.add_edge(*e)
    print("Demo graph built.")
    print("Vertex 1 type:", g.vertex_type(1), "(3 means high-degree)")
    print("BFS distances from 1:", g.bfs(1))
    print("PageRank (top 5):", sorted(((k, round(v, 4)) for k, v in g.pagerank().items()), key=lambda x: -x[1])[:5])
    print("SSSP from 1:", g.sssp(1))
    print("Weak components:", g.connected_components())
    print("Deleting edge 1->3:", g.remove_edge(1, 3))
    print("Vertex 1 degree after deletion:", g.degree(1))


def trace() -> None:  # step-by-step view of one vertex being promoted, resized and demoted
    print("TRACE: TH0=2, TH1=8. Vertex 1 gains edges 2..10, a duplicate, then loses them.\n")  # step-by-step description printed first
    g = GraphTango(th0=2, th1=8, hysteresis=True, on_event=lambda m: print("      *", m))
    steps = [("add", d) for d in range(2, 11)] + [("add", 2)] + [("del", d) for d in range(10, 1, -1)]
    for op, d in steps:
        print(f"{'add_edge' if op == 'add' else 'remove_edge'}(1,{d})")
        r = g.add_edge(1, d) if op == "add" else g.remove_edge(1, d)
        if op == "add" and not r:
            print("      = DUPLICATE: weight updated, nothing added")
        print("   ", g.describe(1))
    print(f"\npromotions={g.promotions} demotions={g.demotions} array_resizes={g.array_resizes}")


def selftest() -> None:  # randomized check of all formats against a plain dict reference model
    """Randomised check of all three formats (both index types) against a plain dict model."""
    rng = random.Random(7)
    makers = {"tango-dict": lambda d: GraphTango(d, 3, 10), "tango-cfh": lambda d: GraphTango(d, 3, 10, index="cfh"),  # every implementation variant under test
              "tango-hyst": lambda d: GraphTango(d, 3, 10, hysteresis=True),
              "adlist": AdListGraph, "hash": HashGraph}
    for name, mk in makers.items():
        for directed in (True, False):
            g, model = mk(directed), {}
            for _ in range(4000):
                u, v = rng.randrange(12), rng.randrange(60)  # few sources, many destinations -> high-degree vertices
                if rng.random() < 0.6:  # 60% inserts, 40% deletes
                    w = float(rng.randrange(1, 9))
                    assert g.add_edge(u, v, w) == (v not in model.get(u, {})), (name, "add")  # return value must match whether the edge was new
                    model.setdefault(u, {})[v] = w
                    if not directed:
                        model.setdefault(v, {})[u] = w
                else:
                    had = v in model.get(u, {})
                    assert g.remove_edge(u, v) == had, (name, "remove")
                    if had:
                        del model[u][v]
                        if not directed:
                            model[v].pop(u, None)
            for u in range(60):
                assert dict(g.neighbors(u)) == model.get(u, {}), (name, directed, u)  # adjacency of each vertex must match the model exactly
            print(f"  ok  {name:11s} directed={directed}")
    print("All self-tests passed.")


# ---------------------------------------------------------------------------------- CLI
def main() -> None:  # command-line entry point
    p = argparse.ArgumentParser(description="GraphTango educational implementation + experiment runner")
    sub = p.add_subparsers(dest="command", required=True)  # subcommands: demo, trace, test, download, benchmark, run, synthetic
    sub.add_parser("demo", help="tiny graph: BFS/PageRank/SSSP/components")
    sub.add_parser("trace", help="watch one vertex change type step by step")
    sub.add_parser("test", help="randomised self-test of every format")
    ds = list(DATASETS)  # dataset names
    for cmd, helptxt in (("download", "download a SNAP dataset"),
                         ("benchmark", "run the experiment on a dataset (alias of run)"),
                         ("run", "download (if needed) and run the full experiment")):
        s = sub.add_parser(cmd, help=helptxt)
        s.add_argument("--dataset", choices=ds + ["all"] if cmd != "download" else ds, required=True)
        s.add_argument("--data-dir", type=Path, default=Path("data"))
        s.add_argument("--limit", type=int, default=200_000 if cmd != "download" else None,
                       help="max edges per dataset (default 200000; use 0 for everything - needs lots of RAM)")
        s.add_argument("--th0", type=int, default=7)
        s.add_argument("--th1", type=int, default=32)
        if cmd != "download":
            s.add_argument("--batch-size", type=int, default=50_000)
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--index", choices=list(INDEX_IMPLS), default="dict",
                           help="Type-3 index: dict (fast) or cfh (paper-style cache-friendly table)")
            s.add_argument("--head", action="store_true",
                           help="use the first --limit lines instead of a random sample (biased: SNAP files are sorted by source)")
            s.add_argument("--csv", help="append results to this CSV")
            s.add_argument("--plot", help="save a bar chart PNG (needs matplotlib)")
    s = sub.add_parser("synthetic", help="offline experiment on generated graphs")
    s.add_argument("--kind", choices=["short", "heavy"], default="heavy")  # synthetic graph shape: heavy-tailed or uniform
    s.add_argument("--vertices", type=int, default=20_000)
    s.add_argument("--edges", type=int, default=200_000)
    s.add_argument("--batch-size", type=int, default=50_000)
    s.add_argument("--th0", type=int, default=7); s.add_argument("--th1", type=int, default=32)
    s.add_argument("--index", choices=list(INDEX_IMPLS), default="dict")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--csv"); s.add_argument("--plot")
    a = p.parse_args()  # parse CLI arguments

    if a.command == "demo":  # dispatch to the chosen subcommand
        demo()
    elif a.command == "trace":
        trace()
    elif a.command == "test":
        selftest()
    elif a.command == "download":
        download_dataset(a.dataset, a.data_dir)
    elif a.command == "synthetic":
        run_experiment(f"synthetic-{a.kind}", synthetic_edges(a.kind, a.vertices, a.edges, a.seed), True,
                       a.batch_size, a.th0, a.th1, a.index, a.seed, a.csv, a.plot)
    else:
        for name in (ds if a.dataset == "all" else [a.dataset]):  # 'all' runs every dataset, otherwise just the chosen one
            path = download_dataset(name, a.data_dir)
            if not a.limit:  # limit=0 means no limit (read the whole file)
                edges = list(parse_edges(path))
            elif a.head:  # first N lines: biased, kept for comparison
                edges = list(parse_edges(path, limit=a.limit))
            else:  # default: uniform random sample of N edges from the whole file
                edges = sample_edges(path, a.limit, a.seed)
            if not edges:
                sys.exit("No edges were parsed. Check the dataset file.")
            plot = a.plot.replace(".png", f"_{name}.png") if a.plot and a.dataset == "all" else a.plot
            run_experiment(name, edges, bool(DATASETS[name]["directed"]), a.batch_size, a.th0, a.th1,
                           a.index, a.seed, a.csv, plot)


if __name__ == "__main__":  # run main() only when executed as a script
    main()

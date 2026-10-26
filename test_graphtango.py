"""Functional tests for the educational GraphTango Python prototype."""  # unit tests for the GraphTango prototype
import math  # math.isclose for float comparison
import unittest  # standard-library test framework
from graphtango_python import GraphTango  # class under test


class GraphTangoTests(unittest.TestCase):  # each test_* method is run as a separate test
    def test_vertex_type_promotes_and_demotes(self):  # degree thresholds: TH0=2, TH1=4
        g = GraphTango(directed=True, th0=2, th1=4)  # small thresholds make type changes easy to trigger
        for dst in (2, 3):
            g.add_edge(1, dst)
        self.assertEqual(g.vertex_type(1), 1)  # degree <= TH0  # degree 2 <= TH0 -> Type 1
        g.add_edge(1, 4)
        self.assertEqual(g.vertex_type(1), 2)  # TH0 < degree <= TH1  # degree 3 -> Type 2
        g.add_edge(1, 5)
        g.add_edge(1, 6)
        self.assertEqual(g.vertex_type(1), 3)  # degree > TH1  # degree 5 > TH1 -> Type 3
        g.remove_edge(1, 6)
        self.assertEqual(g.vertex_type(1), 2)  # degree 4 -> demoted back to Type 2
        g.remove_edge(1, 5)
        self.assertEqual(g.vertex_type(1), 2)
        g.remove_edge(1, 4)
        self.assertEqual(g.vertex_type(1), 1)  # degree 2 -> Type 1 again
        self.assertEqual(g.degree(1), 2)

    def test_duplicate_insert_updates_weight_without_inflating_count(self):  # adding an existing edge only updates its weight
        g = GraphTango()
        self.assertTrue(g.add_edge(1, 2, 3))  # first insert returns True (new edge)
        self.assertFalse(g.add_edge(1, 2, 8))  # duplicate returns False
        self.assertEqual(g.edge_count, 1)  # edge count is not inflated
        self.assertEqual(dict(g.neighbors(1))[2], 8)  # weight was overwritten with the new value

    def test_remove_missing_edge_and_high_degree_index(self):  # Type 3 vertex: hash index must stay consistent after removals
        g = GraphTango(directed=True, th0=1, th1=2)
        for dst in range(2, 7):
            g.add_edge(1, dst)  # five edges with th1=2 -> Type 3
        self.assertEqual(g.vertex_type(1), 3)
        self.assertFalse(g.remove_edge(1, 99))  # removing a nonexistent edge returns False
        self.assertTrue(g.remove_edge(1, 3))  # removing a middle edge (swap-with-last happens)
        self.assertEqual({n for n, _ in g.neighbors(1)}, {2, 4, 5, 6})  # remaining neighbors are correct
        self.assertTrue(g.remove_edge(1, 6))
        self.assertEqual({n for n, _ in g.neighbors(1)}, {2, 4, 5})
        self.assertEqual(g.edge_count, 3)

    def test_undirected_edge_is_symmetric_and_counted_once(self):  # undirected: both directions stored, one logical edge
        g = GraphTango(directed=False)
        self.assertTrue(g.add_edge(1, 2, 2))
        self.assertEqual(g.edge_count, 1)  # counted once despite two stored directions
        self.assertEqual(dict(g.neighbors(1))[2], 2)
        self.assertEqual(dict(g.neighbors(2))[1], 2)  # reverse direction has the same weight
        self.assertTrue(g.remove_edge(1, 2))  # removal clears both directions
        self.assertEqual(g.edge_count, 0)
        self.assertEqual(g.degree(1), 0)
        self.assertEqual(g.degree(2), 0)

    def test_bfs_and_weighted_sssp(self):  # BFS hop distances and Dijkstra weighted distances
        g = GraphTango()
        g.add_edge(1, 2, 4)
        g.add_edge(1, 3, 1)
        g.add_edge(3, 2, 1)
        g.add_edge(2, 4, 2)
        self.assertEqual(g.bfs(1), {1: 0, 2: 1, 3: 1, 4: 2})  # hop distances from vertex 1
        dist = g.sssp(1)
        self.assertEqual(dist[2], 2)  # cheapest 1->2 is via 3: 1+1 = 2
        self.assertEqual(dist[4], 4)  # 1->3->2->4 = 1+1+2 = 4

    def test_sssp_rejects_negative_weight_when_encountered(self):  # negative weights must raise ValueError
        g = GraphTango()
        g.add_edge(1, 2, -1)
        with self.assertRaises(ValueError):
            g.sssp(1)

    def test_pagerank_is_normalized(self):  # PageRank scores sum to 1
        g = GraphTango()
        g.add_edge(1, 2)
        g.add_edge(2, 1)
        g.add_edge(2, 3)
        ranks = g.pagerank(iterations=100)
        self.assertEqual(set(ranks), {1, 2, 3})
        self.assertTrue(math.isclose(sum(ranks.values()), 1.0, rel_tol=1e-7, abs_tol=1e-7))  # allow tiny floating-point error

    def test_weak_connected_components_for_directed_graph(self):  # directed edges 1->2 and 3->4 form two weak components
        g = GraphTango(directed=True)
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        labels = g.connected_components()
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])  # same component for 3 and 4
        self.assertNotEqual(labels[1], labels[3])  # different components


if __name__ == "__main__":  # allow running this file directly
    unittest.main(verbosity=2)

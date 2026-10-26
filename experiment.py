from graphtango_python import GraphTango  # import the GraphTango class from the main module

graph = GraphTango(directed=True, th0=2, th1=4)  # directed graph; Type 1 up to degree 2, Type 2 up to degree 4, Type 3 above

graph.add_edge(1, 2)  # add edge 1 -> 2
graph.add_edge(1, 3)  # add edge 1 -> 3 (degree 2 = still Type 1)

print("Degree:", graph.degree(1))  # out-degree of vertex 1
print("Vertex type:", graph.vertex_type(1))  # current representation type of vertex 1

graph.add_edge(1, 4)  # third edge pushes degree above TH0 -> promoted to Type 2

print("Degree after insertion:", graph.degree(1))  # degree should now be 3
print("Vertex type after insertion:", graph.vertex_type(1))  # vertex type should now be 2

graph.remove_edge(1, 4)  # remove the edge again

print("Degree after deletion:", graph.degree(1))  # degree back to 2
print("Vertex type after deletion:", graph.vertex_type(1))  # vertex demoted back to Type 1

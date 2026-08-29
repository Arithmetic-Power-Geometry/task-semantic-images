import networkx as nx
import numpy as np
from tasksemantic.core import (
    cardinality_image_bipartite, exact_cardinality_image, minimum_vertex_cover_bipartite,
    verify_proper_affine_hyperplane, proper_affine_observer, affine_image_bruteforce,
    minimum_cover_partition_image_bruteforce, constrained_min_cover_state_bruteforce,
    compose_affine_coset, affine_image_from_coset, exact_ordered_bdd_size_graph_cnf,
    task_replacement_decision,
)

def test_cycle_cardinality_exact():
    G=nx.cycle_graph(8); assert cardinality_image_bipartite(G).states==exact_cardinality_image(G).states

def test_bipartite_cover_is_cover():
    G=nx.complete_bipartite_graph(3,5); C=minimum_vertex_cover_bipartite(G); assert len(C)==3

def test_cardinality_interval():
    G=nx.path_graph(8); r=cardinality_image_bipartite(G); assert r.states==tuple(range(r.witness_count,9))

def test_affine_hyperplane():
    G=nx.cycle_graph(10); assert verify_proper_affine_hyperplane(G,4)['verified']

def test_affine_observer_dense():
    G=nx.cycle_graph(12); _,_,A,_=proper_affine_observer(G,4); assert int(A.sum(axis=1).min())>=8

def test_affine_bruteforce_nonempty():
    G=nx.cycle_graph(8); _,_,A,c=proper_affine_observer(G,4); assert len(affine_image_bruteforce(G,A,c).states)==8

def test_barrier_image_minimum_size():
    G=nx.cycle_graph(8); left={0,2,4,6}; r=minimum_cover_partition_image_bruteforce(G,left); assert r.witness_count==4

def test_barrier_threshold():
    G=nx.path_graph(6); left={0,2,4}; assert constrained_min_cover_state_bruteforce(G,left,3,3)

def test_affine_composition_identity():
    offset=np.array([1,0],dtype=np.uint8); basis=np.array([[1],[1]],dtype=np.uint8); B=np.eye(2,dtype=np.uint8); d=np.zeros(2,dtype=np.uint8); o2,b2=compose_affine_coset(offset,basis,B,d); assert affine_image_from_coset(o2,b2)==affine_image_from_coset(offset,basis)

def test_bdd_terminal_count():
    G=nx.Graph(); G.add_nodes_from(range(3)); assert exact_ordered_bdd_size_graph_cnf(G)==2

def test_bdd_nontrivial():
    G=nx.path_graph(6); assert exact_ordered_bdd_size_graph_cnf(G)>2

def test_task_replacement():
    assert task_replacement_decision([2,4,6],lambda y:y==4)

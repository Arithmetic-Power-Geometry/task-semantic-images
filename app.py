from __future__ import annotations
import json, math
import gradio as gr
import networkx as nx
from tasksemantic.core import cardinality_image_bipartite, verify_proper_affine_hyperplane, minimum_cover_partition_image_bruteforce, exact_ordered_bdd_size_graph_cnf
from tasksemantic.experiments import make_bipartite_cycle, make_random_bipartite

def graph(family,n,p,seed):
    n=int(n); seed=int(seed)
    return make_bipartite_cycle(n) if family=='cycle' else make_random_bipartite(n,float(p),seed)

def cardinality(family,n,p,seed):
    G=graph(family,n,p,seed); r=cardinality_image_bipartite(G)
    return {'n':G.number_of_nodes(),'m':G.number_of_edges(),'minimum_cover_tau':r.witness_count,'reachable_cardinalities':list(r.states),'state_count':len(r.states),'log2_assignment_to_image_ratio':G.number_of_nodes()-math.log2(max(len(r.states),1))}

def affine(family,n,p,seed,k):
    G=graph(family,n,p,seed); return verify_proper_affine_hyperplane(G,int(k))

def barrier(n,p,seed):
    G=make_random_bipartite(int(n),float(p),int(seed)); color=nx.algorithms.bipartite.color(G); left={v for v,c in color.items() if c==0}; r=minimum_cover_partition_image_bruteforce(G,left)
    return {'n':G.number_of_nodes(),'m':G.number_of_edges(),'minimum_cover':r.witness_count,'reachable_partition_states':list(r.states),'reachable_count':len(r.states),'possible_count':(len(left)+1)*(G.number_of_nodes()-len(left)+1)}

def bdd(family,n,p,seed):
    G=graph(family,n,p,seed); tsi=cardinality_image_bipartite(G); return {'n':G.number_of_nodes(),'m':G.number_of_edges(),'exact_ordered_bdd_nodes':exact_ordered_bdd_size_graph_cnf(G),'cardinality_tsi_states':len(tsi.states),'note':'Finite fixed-order OBDD diagnostic; the manuscript DNNF lower bound is theorem-based.'}

with gr.Blocks(title='Task-Semantic Images Lab') as demo:
    gr.Markdown('# Task-Semantic Images Lab\nExact finite diagnostics for cardinality images, affine semantic collapse, realizability barriers, and OBDD-vs-TSI state counts.')
    with gr.Tab('Cardinality image'):
        fam=gr.Dropdown(['cycle','random'],value='cycle',label='Graph family'); n=gr.Slider(4,400,20,step=2,label='Variables'); p=gr.Slider(.02,.8,.2,label='Random edge probability'); seed=gr.Number(2026,label='Seed'); out=gr.JSON(); gr.Button('Compute').click(cardinality,[fam,n,p,seed],out)
    with gr.Tab('Affine theorem check'):
        fam2=gr.Dropdown(['cycle','random'],value='cycle',label='Graph family'); n2=gr.Slider(4,20,12,step=2,label='Variables (exhaustive)'); p2=gr.Slider(.02,.8,.2,label='Random edge probability'); seed2=gr.Number(2026,label='Seed'); k=gr.Slider(2,8,4,step=2,label='Observer dimension k'); out2=gr.JSON(); gr.Button('Verify').click(affine,[fam2,n2,p2,seed2,k],out2)
    with gr.Tab('Semantic-resolution barrier'):
        n3=gr.Slider(4,20,12,step=2,label='Variables (exhaustive)'); p3=gr.Slider(.05,.8,.3,label='Edge probability'); seed3=gr.Number(2026,label='Seed'); out3=gr.JSON(); gr.Button('Enumerate minimum-cover image').click(barrier,[n3,p3,seed3],out3)
    with gr.Tab('Finite OBDD diagnostic'):
        fam4=gr.Dropdown(['cycle','random'],value='cycle',label='Graph family'); n4=gr.Slider(4,24,16,step=2,label='Variables'); p4=gr.Slider(.02,.8,.25,label='Random edge probability'); seed4=gr.Number(2026,label='Seed'); out4=gr.JSON(); gr.Button('Compare').click(bdd,[fam4,n4,p4,seed4],out4)

if __name__=='__main__': demo.launch()

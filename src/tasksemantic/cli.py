from __future__ import annotations
import argparse, json
import networkx as nx
from .core import cardinality_image_bipartite, verify_proper_affine_hyperplane, minimum_cover_partition_image_bruteforce, exact_ordered_bdd_size_graph_cnf
from .experiments import reproduce, make_bipartite_cycle, make_random_bipartite

def main():
    p=argparse.ArgumentParser(prog='tasksemantic',description='Task-Semantic Images reproducibility CLI')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('reproduce')
    c=sub.add_parser('cardinality'); c.add_argument('--n',type=int,default=20); c.add_argument('--family',choices=['cycle','random'],default='cycle'); c.add_argument('--p',type=float,default=.2); c.add_argument('--seed',type=int,default=2026)
    a=sub.add_parser('affine'); a.add_argument('--n',type=int,default=12); a.add_argument('--k',type=int,default=4)
    b=sub.add_parser('barrier'); b.add_argument('--n',type=int,default=12); b.add_argument('--p',type=float,default=.3); b.add_argument('--seed',type=int,default=2026)
    o=sub.add_parser('bdd'); o.add_argument('--n',type=int,default=16); o.add_argument('--family',choices=['cycle','random'],default='cycle')
    args=p.parse_args()
    if args.cmd=='reproduce': print(json.dumps(reproduce(),indent=2)); return
    G=make_bipartite_cycle(args.n) if getattr(args,'family','cycle')=='cycle' else make_random_bipartite(args.n,getattr(args,'p',.2),getattr(args,'seed',2026))
    if args.cmd=='cardinality':
        r=cardinality_image_bipartite(G); print(json.dumps({'n':G.number_of_nodes(),'m':G.number_of_edges(),'tau':r.witness_count,'states':r.states},indent=2)); return
    if args.cmd=='affine': print(json.dumps(verify_proper_affine_hyperplane(G,args.k),indent=2)); return
    if args.cmd=='barrier':
        color=nx.algorithms.bipartite.color(G); left={v for v,c in color.items() if c==0}; r=minimum_cover_partition_image_bruteforce(G,left); print(json.dumps({'minimum_cover':r.witness_count,'states':r.states},indent=2)); return
    if args.cmd=='bdd': print(json.dumps({'nodes':exact_ordered_bdd_size_graph_cnf(G)},indent=2)); return

if __name__=='__main__': main()

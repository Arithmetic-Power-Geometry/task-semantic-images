from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .core import (
    affine_image_bruteforce,
    affine_image_from_coset,
    cardinality_image_bipartite,
    compose_affine_coset,
    exact_cardinality_image,
    exact_ordered_bdd_size_graph_cnf,
    minimum_cover_partition_image_bruteforce,
    proper_affine_observer,
    verify_proper_affine_hyperplane,
)

ROOT = Path(__file__).resolve().parents[2]


def make_bipartite_cycle(n: int) -> nx.Graph:
    if n % 2:
        n += 1
    return nx.cycle_graph(n)


def make_bipartite_grid(rows: int, cols: int) -> nx.Graph:
    G = nx.grid_2d_graph(rows, cols)
    return nx.convert_node_labels_to_integers(G)


def make_random_bipartite(n: int, p: float, seed: int) -> nx.Graph:
    if n % 2:
        n += 1
    return nx.algorithms.bipartite.random_graph(n // 2, n // 2, p, seed=seed)


def run_cardinality_scaling(ns=(8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256), outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        G = make_bipartite_cycle(n)
        t0 = time.perf_counter(); result = cardinality_image_bipartite(G); dt = time.perf_counter() - t0
        rows.append({
            "family": "cycle", "n": n, "m": G.number_of_edges(), "tau": result.witness_count,
            "image_states": len(result.states), "codomain_states": n + 1,
            "full_assignment_log2": n, "tsi_state_log2": math.log2(max(len(result.states), 1)),
            "runtime_ms": 1000 * dt,
        })
    df = pd.DataFrame(rows); df.to_csv(outdir / "cardinality_scaling.csv", index=False); return df


def run_random_bipartite(ns=(20, 40, 60, 80, 100, 150, 200, 300, 400), seeds=(11, 29, 47, 83, 101), p=0.10, outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        for seed in seeds:
            G = make_random_bipartite(n, p=p, seed=seed)
            t0 = time.perf_counter(); result = cardinality_image_bipartite(G); dt = time.perf_counter() - t0
            rows.append({
                "n": n, "seed": seed, "p": p, "m": G.number_of_edges(), "tau": result.witness_count,
                "image_states": len(result.states), "codomain_states": n + 1,
                "log2_assignment_to_image_ratio": n - math.log2(max(len(result.states), 1)),
                "runtime_ms": 1000 * dt,
            })
    df = pd.DataFrame(rows); df.to_csv(outdir / "random_bipartite.csv", index=False); return df


def run_cardinality_exact_checks(ns=(4, 6, 8, 10, 12, 14, 16, 18), outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        G = make_bipartite_cycle(n)
        t0 = time.perf_counter(); brute = exact_cardinality_image(G); brute_ms = 1000*(time.perf_counter()-t0)
        t0 = time.perf_counter(); fast = cardinality_image_bipartite(G); fast_ms = 1000*(time.perf_counter()-t0)
        rows.append({
            "n": n, "brute_states": len(brute.states), "matching_states": len(fast.states),
            "brute_tau": brute.witness_count, "matching_tau": fast.witness_count,
            "verified": brute.states == fast.states, "brute_ms": brute_ms, "matching_ms": fast_ms,
        })
    df = pd.DataFrame(rows); df.to_csv(outdir / "cardinality_exact_checks.csv", index=False); return df


def run_affine_verification(ns=(6, 8, 10, 12, 14, 16, 18), outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        G = make_bipartite_cycle(n)
        k = max(2, int(math.log2(n))); k += k % 2
        t0 = time.perf_counter(); row = verify_proper_affine_hyperplane(G, k); row["runtime_ms"] = 1000*(time.perf_counter()-t0)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(outdir / "affine_verification.csv", index=False); return df


def run_affine_composition(ns=(8, 10, 12, 14, 16), stages=4, seed=2026, outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        G = make_bipartite_cycle(n)
        k = max(2, int(math.log2(n))); k += k % 2
        nodes, I, A, c = proper_affine_observer(G, k)
        brute0 = set(affine_image_bruteforce(G, A, c).states)
        basis = np.zeros((k, k-1), dtype=np.uint8)
        for i in range(k-1): basis[i, i] = 1; basis[k-1, i] = 1
        # The theorem image is the odd-parity hyperplane: e1 + even-parity subspace.
        offset = np.zeros(k, dtype=np.uint8); offset[0] = 1
        predicted0 = set(affine_image_from_coset(offset, basis))
        assert brute0 == predicted0
        current_brute = brute0
        current_offset, current_basis = offset, basis
        for stage in range(1, stages+1):
            r = max(2, k - (stage % 2))
            B = rng.integers(0, 2, size=(r, len(current_offset)), dtype=np.uint8)
            d = rng.integers(0, 2, size=r, dtype=np.uint8)
            current_brute = {tuple(int(v) for v in ((B @ np.asarray(y,dtype=np.uint8)+d)%2)) for y in current_brute}
            current_offset, current_basis = compose_affine_coset(current_offset, current_basis, B, d)
            predicted = set(affine_image_from_coset(current_offset, current_basis))
            rows.append({"n":n,"k0":k,"stage":stage,"observed_states":len(current_brute),"predicted_states":len(predicted),"verified":current_brute==predicted})
    df = pd.DataFrame(rows); df.to_csv(outdir / "affine_composition.csv", index=False); return df


def run_barrier_finite(ns=(6, 8, 10, 12, 14, 16), seeds=(7,19,31), p=0.30, outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in ns:
        for seed in seeds:
            G = make_random_bipartite(n, p=p, seed=seed)
            color = nx.algorithms.bipartite.color(G)
            left = {v for v,c in color.items() if c == 0}
            t0 = time.perf_counter(); image = minimum_cover_partition_image_bruteforce(G,left); dt = time.perf_counter()-t0
            codomain = (len(left)+1)*(n-len(left)+1)
            rows.append({
                "n":n,"seed":seed,"m":G.number_of_edges(),"minimum_cover":image.witness_count,
                "reachable_partition_states":len(image.states),"codomain_states":codomain,
                "state_fraction":len(image.states)/codomain,"enumeration_ms":1000*dt,
            })
    df = pd.DataFrame(rows); df.to_csv(outdir / "barrier_finite.csv", index=False); return df


def run_bdd_diagnostics(ns=(6, 8, 10, 12, 14, 16, 18, 20), outdir=None):
    outdir = Path(outdir or ROOT / "results"); outdir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for n in ns:
        families = {
            "cycle": make_bipartite_cycle(n),
            "random_bipartite": make_random_bipartite(n, p=0.25, seed=100+n),
        }
        for family,G in families.items():
            t0=time.perf_counter(); bdd=exact_ordered_bdd_size_graph_cnf(G); dt=time.perf_counter()-t0
            tsi=cardinality_image_bipartite(G)
            rows.append({"family":family,"n":n,"m":G.number_of_edges(),"ordered_bdd_nodes":bdd,"cardinality_tsi_states":len(tsi.states),"bdd_runtime_ms":1000*dt})
    df=pd.DataFrame(rows); df.to_csv(outdir/'bdd_diagnostics.csv',index=False); return df


def make_figures(outdir=None, figdir=None):
    outdir=Path(outdir or ROOT/'results'); figdir=Path(figdir or ROOT/'figures'); figdir.mkdir(parents=True,exist_ok=True)

    d=pd.read_csv(outdir/'cardinality_scaling.csv')
    plt.figure(figsize=(6.4,4.2)); plt.plot(d.n,d.full_assignment_log2,marker='o',label='log2 full assignments'); plt.plot(d.n,d.tsi_state_log2,marker='s',label='log2 cardinality TSI states'); plt.xlabel('Variables n'); plt.ylabel('log2 state count'); plt.title('Complete assignment space vs task-semantic image'); plt.legend(); plt.tight_layout(); plt.savefig(figdir/'fig_state_gap.pdf'); plt.savefig(figdir/'fig_state_gap.png',dpi=180); plt.close()

    d=pd.read_csv(outdir/'affine_verification.csv')
    plt.figure(figsize=(6.4,4.2)); plt.plot(d.n,d.observed_states,marker='o',label='Observed'); plt.plot(d.n,d.expected_states,linestyle='--',label='Theorem'); plt.xlabel('Variables n'); plt.ylabel('Affine output states'); plt.title('Proper-affine semantic image: exhaustive verification'); plt.legend(); plt.tight_layout(); plt.savefig(figdir/'fig_affine_verify.pdf'); plt.savefig(figdir/'fig_affine_verify.png',dpi=180); plt.close()

    d=pd.read_csv(outdir/'barrier_finite.csv')
    agg=d.groupby('n',as_index=False).agg(reachable=('reachable_partition_states','mean'),codomain=('codomain_states','mean'),runtime=('enumeration_ms','mean'))
    plt.figure(figsize=(6.4,4.2)); plt.plot(agg.n,agg.codomain,marker='o',label='Possible semantic states'); plt.plot(agg.n,agg.reachable,marker='s',label='Reachable minimum-cover states'); plt.xlabel('Variables n'); plt.ylabel('State count'); plt.title('Finite semantic-resolution barrier diagnostic'); plt.legend(); plt.tight_layout(); plt.savefig(figdir/'fig_barrier_states.pdf'); plt.savefig(figdir/'fig_barrier_states.png',dpi=180); plt.close()

    d=pd.read_csv(outdir/'bdd_diagnostics.csv')
    plt.figure(figsize=(6.4,4.2));
    for family,g in d.groupby('family'):
        plt.plot(g.n,g.ordered_bdd_nodes,marker='o',label=f'{family}: OBDD nodes')
    plt.plot(d[d.family=='cycle'].n,d[d.family=='cycle'].cardinality_tsi_states,linestyle='--',label='cycle: cardinality TSI states')
    plt.xlabel('Variables n'); plt.ylabel('Exact finite state/node count'); plt.title('Finite OBDD diagnostic vs task-semantic states'); plt.legend(); plt.tight_layout(); plt.savefig(figdir/'fig_bdd_vs_tsi.pdf'); plt.savefig(figdir/'fig_bdd_vs_tsi.png',dpi=180); plt.close()

    d=pd.read_csv(outdir/'affine_composition.csv')
    agg=d.groupby('stage',as_index=False).agg(observed=('observed_states','mean'),predicted=('predicted_states','mean'))
    plt.figure(figsize=(6.4,4.2)); plt.plot(agg.stage,agg.observed,marker='o',label='Observed image'); plt.plot(agg.stage,agg.predicted,linestyle='--',label='Affine prediction'); plt.xlabel('Composition stage'); plt.ylabel('Mean reachable semantic states'); plt.title('Exact affine composition remains closed'); plt.legend(); plt.tight_layout(); plt.savefig(figdir/'fig_affine_composition.pdf'); plt.savefig(figdir/'fig_affine_composition.png',dpi=180); plt.close()


def reproduce(outdir=None,figdir=None):
    outdir=Path(outdir or ROOT/'results'); figdir=Path(figdir or ROOT/'figures')
    d1=run_cardinality_scaling(outdir=outdir)
    d2=run_random_bipartite(outdir=outdir)
    d3=run_cardinality_exact_checks(outdir=outdir)
    d4=run_affine_verification(outdir=outdir)
    d5=run_affine_composition(outdir=outdir)
    d6=run_barrier_finite(outdir=outdir)
    d7=run_bdd_diagnostics(outdir=outdir)
    make_figures(outdir,figdir)
    summary={
        'cardinality_rows':len(d1),'random_rows':len(d2),'exact_check_rows':len(d3),
        'affine_rows':len(d4),'composition_rows':len(d5),'barrier_rows':len(d6),'bdd_rows':len(d7),
        'cardinality_all_verified':bool(d3.verified.all()),'affine_all_verified':bool(d4.verified.all()),
        'composition_all_verified':bool(d5.verified.all()),
        'max_random_n':int(d2.n.max()),'max_bdd_n':int(d7.n.max()),
    }
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    print(json.dumps(reproduce(),indent=2))

# Nanosatellite Swarm Resilience

This project explores the **resilience and robustness** of nanosatellite swarms through simulation and advanced graph partitioning techniques. It was developed as part of an academic research project in the **Telecommunications & Networks** track at INP ENSEEIHT.

---

## Project Context

Modern satellite communication systems increasingly rely on **swarms of nanosatellites** to achieve global coverage, redundancy, and mission flexibility. However, these dynamic, decentralized networks are also vulnerable to failures — whether random (e.g. battery drain) or targeted (e.g. node failures).

Inspired by the **divide-and-conquer paradigm**, this project investigates how **graph partitioning algorithms** can be used to **increase the fault tolerance** of satellite swarms by structuring their communication topology into more resilient and self-contained clusters.

---

## Inspiration and Research Background

This work is **inspired by the PhD thesis of Evelyne Akopyan**, who introduced a simulation framework for analyzing topological resilience in nanosatellite swarms using graph theory.
---

## Simulation Methodology

The simulation runs over 1000 time steps with 100 satellites in motion. At each step:

1. A communication graph is built based on spatial proximity.
2. A partitioning algorithm divides the graph into clusters.
3. Key resilience metrics are computed.
4. A fault scenario (e.g., targeted node removal) is simulated at `t = to vary`.
5. Metrics are re-evaluated post-failure.
6. Results are logged for comparison.

Simulation is controlled through the following components:

- `simulation.ipynb` — main notebook to run the simulation and plots
- `swarm_sim.py` — builds the temporal graphs and applies clustering
- `metrics.py` — evaluates robustness, resilience, routing cost, etc.

---

##  Technologies Used

| Component       | Tool / Language        |
|----------------|------------------------|
| Language        | Python 3               |
| Simulation      | Jupyter Notebook       |
| Graph Algorithms| Custom partitioning logic (FFD, MIRW, RND) |
| Data analysis   | pandas, NumPy          |
| Visualization   | matplotlib             |
| Output logs     | CSV files              |

---

## Resilience Strategies

We compare the following strategies:

- **Random Node Division (RND)** — fast, unstructured clustering
- **Forest Fire Division (FFD)** — local propagation from seed nodes
- **MIRW (Multiple Independent Random Walks)** — community discovery via random walks
- **K-Means** — baseline geometric clustering (for comparison)

Metrics evaluated include:

- Path redundancy
- Path disparity
- Node criticality
- Flow robustness
- Routing cost
- Cluster equity (RMSE)

---
##  How to Use

1. Clone the repository:
```bash
git clone https://github.com/aei6973/Nanosatellite-Swarm-Resilience.git
cd Nanosatellite-Swarm-Resilience

2. Launch the notebook:
```bash
jupyter notebook simulation.ipynb

3. Edit simulation parameters (number of nodes, failure type, clustering method...) in the notebook.

4. Run the cells and analyze visualizations and metrics.
---

## Authors
Aicha ELIDRISSI
Yasmina Abou-El-Abbas

Supervised by: Dr. Riadh Dhaou
INP ENSEEIHT – Department of Computer Science and Telecommunications


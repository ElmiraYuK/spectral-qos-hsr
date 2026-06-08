# Spectral QoS Controller on a High-Speed-Rail Corridor

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elmira-k/spectral-qos-hsr/blob/main/VSM_radio_map_QoS_experiment_EN.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Companion code for the paper on a periodic **M/M(t)/1** model of an on-board traffic aggregator in a 5G/6G **high-speed-rail (HSR)** network. The service rate `μ(t)` is shaped by the moving train passing a regular lattice of base stations along a **trans-Alpine corridor (Paris–Milan)**; sparse cellular coverage makes `μ(t)` quasi-periodic with occasional deep fades ("coverage gaps").

**Author.** El'mira Yu. Kalimulina — Kharkevich Institute for Information Transmission Problems of the Russian Academy of Sciences (IITP RAS); Lomonosov Moscow State University.

## What the code does

* Builds a **geo-anchored** service-rate profile `μ(t)` with **five coverage gaps** (three short full outages, two partial fades) at cell edges.
* Solves the time-periodic Kolmogorov equations by a **harmonic-balance / Fourier–Galerkin** spectral method and **validates it against Monte Carlo** (relative error within about **1.6 %**, and ≈0.1 % at ρ = 0.7; exact `ρ/(1−ρ)` for constant `μ`).
* Compares **six admission controllers** — DropTail, Static, Reactive (delayed feedback), Radio-map look-ahead, the proposed **Spectral+map** controller (its control kernel is the harmonic-balance solution itself), and a lightweight tabular Q-learning.
* Reports the headline results:
  * on the **periodic backbone**, the spectral controller attains the lowest 99th-percentile delay, with a widening margin (**−18 %** vs Reactive at ρ = 0.7, **−28 %** at ρ = 0.85);
  * on the **full profile** with coverage gaps, **Spectral+map** gives the lowest mean and tail delay (about **−27 % / −22 %** vs Reactive and **−54 % / −42 %** vs DropTail at ρ = 0.85; overflow 2.2 %→0.1 %), at a deliberate goodput cost (0.84 vs 0.94).
* Studies **robustness across speed** (180–420 km/h) and the **proactive pre-draining** mechanism (mean queue folded over track position).

> **Honest framing.** This is a delay-QoS trade-off, not a win on every metric: the proactive layer admits a little less traffic in exchange for a much shorter tail. The "proactive vs reactive" advantage depends on the stated feedback delay `τ_fb = 0.8 s`. Q-learning is a lightweight comparator, not a tuned DQN.

## Repository layout

```
.
├── README.md
├── LICENSE                                  # MIT
├── requirements.txt
├── CITATION.cff
├── PUBLISHING.md                            # step-by-step: Colab → GitHub → Zenodo
├── vsm_spectral_qos_experiment.py           # standalone script (headless; writes PNG/HTML)
└── VSM_radio_map_QoS_experiment_EN.ipynb    # annotated notebook (theory + figures)
```

## Quickstart

**Colab (no install).** Click the *Open in Colab* badge above, then *Runtime → Run all*. The first cell installs the dependencies.

**Local — notebook.**
```bash
pip install -r requirements.txt
jupyter lab notebooks/VSM_radio_map_QoS_experiment_EN.ipynb
```

**Local — standalone script (headless).**
```bash
pip install -r requirements.txt
python vsm_spectral_qos_experiment.py
```
The script uses the non-interactive `Agg` backend and writes the figures as `fig_07.png … fig_11.png`, the static coverage maps as `coverage_static.png` and `coverage_basemap.png`, and the interactive map as `coverage_map.html`. Reproducibility seed: `SEED = 7`.

## Reproducibility note

Per the paper's data statement, **the corridor geometry does not enter the service-rate generation** — it only positions the map. The public corridor (Paris–Milan, ≈645 km, ≈170 base stations) is marginally shorter than the anonymised corridor reported in the paper (≈654 km, 172 base stations). Consequently the **periodic-backbone results and the solver validation are identical** (they depend only on cell spacing and train speed, giving `T0 = 45.6 s`), while the full-profile figures may differ by about 1 %.

## Citing

If you use this software, please cite **both** the paper and this software record (see `CITATION.cff`):

* *Paper:* An Edge-Deployable Spectral QoS Controller for Periodic Traffic Aggregation in 5G/6G Mobile High-Speed Platforms. *(journal / DOI to be added on acceptance.)*
* *Software:* El'mira Yu. Kalimulina, *Spectral QoS Controller on a High-Speed-Rail Corridor* (2026), Zenodo, DOI: `10.5281/zenodo.XXXXXXX`.

## Related

The underlying on-board admission/aggregation method is protected by patent **RU 2843669 C1**. This repository releases the research source code for reproducibility under the MIT licence; the licence covers the source code.

## Licence

MIT — see [`LICENSE`](LICENSE). *(If IITP RAS is the institutional rights holder for your software registrations, you may wish to match the copyright line accordingly.)*

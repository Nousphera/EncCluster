"""Aggregation methods: the full EncCluster secure path and the three baselines it
ablates against. All expose aggregate(...) returning a new state dict."""

from . import fedavg, cluster, bfuse, enccluster

METHODS = {"fedavg": fedavg, "cluster": cluster, "bfuse": bfuse, "enccluster": enccluster}

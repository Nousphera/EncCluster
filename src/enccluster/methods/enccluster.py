# Modified 2026-10-15: coordinate-labelled decentralized FE; see REPRODUCIBILITY.md.
from enccluster.clustering import weight_clustering, unflatten_state, aggregate_buffers
from enccluster.filters import encode_mapping, reconstruct_mapping
from enccluster.coordinate_fe import setup


def aggregate(states, sizes, kappa=16, bpe=8, seed=42, precision=3, max_abs=8.0, **_):
    """Train/cluster/BF numerical path with client-side coordinate encryption.

    Each client queries its own filter before encryption. This gives the same
    reconstructed clustered vector, while avoiding scalar-key/label reuse. It
    costs d ciphertexts rather than kappa; it is not the paper's compressed FE.
    """
    if len(states) != len(sizes):
        raise ValueError("Each client must supply a sample count")
    clustered = [weight_clustering(sd,kappa=kappa,seed=seed) for sd in states]
    keys, shapes = clustered[0][2:]
    if any(c[2] != keys or c[3] != shapes for c in clustered):
        raise ValueError("Clients must use identical state keys and shapes")
    d = len(clustered[0][1])
    clients, server = setup(sizes,dimension=d,precision=precision,max_abs=max_abs)
    messages = []
    for client,(Z,P,_,_) in zip(clients,clustered):
        bf = encode_mapping(P,bpe=bpe)
        reconstructed = reconstruct_mapping(bf,d,kappa)
        messages.append(client.encrypt(Z[reconstructed],round_id=1))
    flat = server.aggregate(messages,round_id=1)
    out = unflatten_state(flat,keys,shapes)
    out.update(aggregate_buffers(states,sizes))
    return out

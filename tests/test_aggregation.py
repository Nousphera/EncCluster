import numpy as np
import torch
from enccluster.clustering import weight_clustering
from enccluster.filters import encode_mapping,reconstruct_mapping
from enccluster.methods.enccluster import aggregate

def test_cluster_filter_encryption_matches_fixed_point_oracle():
    states=[{'weight':torch.tensor([-.3,.5,.1,.8])},{'weight':torch.tensor([.2,-.4,.8,.9])}]
    sizes=[2,5]; expanded=[]
    for state in states:
        z,p,_,_=weight_clustering(state,kappa=2,seed=3)
        mapping=reconstruct_mapping(encode_mapping(p,bpe=16),len(p),2)
        expanded.append(np.rint(z[mapping]*100)/100)
    expected=sum(x*w for x,w in zip(expanded,sizes))/sum(sizes)
    np.testing.assert_allclose(aggregate(states,sizes,kappa=2,bpe=16,precision=2,seed=3)['weight'],expected,atol=1e-7)

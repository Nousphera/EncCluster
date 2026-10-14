import json
from enccluster.train import main,partition

def test_two_round_training_writes_real_metrics(tmp_path):
    output=tmp_path/'run.json'
    result=main(['--output',str(output)])
    assert len(result['history'])==2
    assert all(0<=r['accuracy']<=1 for r in result['history'])
    assert json.loads(output.read_text())['configuration']['seed']==42
    assert result['benchmark_reproduction'] is False

def test_dirichlet_partitions_preserve_disjoint_sample_identity():
    subsets=partition([0]*50+[1]*50,clients=3,alpha=10,seed=7)
    flat=[i for subset in subsets for i in subset]
    assert sorted(flat)==list(range(100)) and len(set(flat))==100


def test_output_refuses_overwrite(tmp_path):
    import pytest
    output=tmp_path/'old.json'
    output.write_text('preserve')
    with pytest.raises(FileExistsError):
        main(['--output',str(output)])
    assert output.read_text()=='preserve'


def test_checkpoint_is_loadable_and_matches_report(tmp_path):
    import torch
    output=tmp_path/'new.json'
    result=main(['--output',str(output),'--method','fedavg','--rounds','1'])
    checkpoint=torch.load(output.with_suffix('.pt'),weights_only=True,map_location='cpu')
    assert checkpoint['configuration']==result['configuration']
    assert checkpoint['model']
    assert result['environment']['torch']

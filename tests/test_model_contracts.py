"""Paper architecture and input contracts; no external weights are downloaded."""
from argparse import Namespace
import pytest
import torch
from enccluster.train import ImageModel, load_data, main


def test_synthetic_inputs_use_the_default_paper_image_shape():
    args = Namespace(dataset="synthetic", model="resnet20", seed=7,
                     clients=2, samples_per_client=2)
    train, test, partitions, labels, classes = load_data(args)
    assert train.tensors[0].shape == (4, 3, 32, 32)
    assert test.tensors[0].shape[1:] == (3, 32, 32)
    assert classes == 3 and len(partitions) == 2


def test_only_paper_models_are_accepted():
    with pytest.raises(SystemExit):
        main(["--model", "unsupported_model", "--rounds", "1", "--method", "fedavg"])


def test_resnet20_block_preserves_the_shortcut_when_residual_is_zero():
    from enccluster.architectures.resnet20 import BasicBlock
    block = BasicBlock(16, 16)
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv2.weight.zero_()
    image = torch.rand(2, 16, 8, 8)
    torch.testing.assert_close(block(image), image)


@pytest.mark.parametrize("name", ['resnet20', 'convnext', 'convmixer256_8', 'mobilenet_v2'])
def test_paper_model_forward_backward_preserves_full_backbone(name):
    model = ImageModel(name, 10)
    images = torch.rand(2, 3, 32, 32)
    output = model(images)
    assert output.shape == (2, 10) and torch.isfinite(output).all()
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert next(model.parameters()).grad.abs().sum() > 0
    backbone = model.body.backbone
    if name == "resnet20":
        assert [len(getattr(backbone, f"layer{i}")) for i in range(1, 4)] == [3, 3, 3]
        assert sum(isinstance(m, torch.nn.Conv2d) for m in backbone.modules()) == 19
    elif name == "convmixer256_8":
        assert sum(isinstance(m, torch.nn.Conv2d) and m.groups == 256 for m in backbone.modules()) == 8
        assert backbone[0].kernel_size == (2, 2) and backbone[0].out_channels == 256
    elif name == "convnext":
        assert [len(getattr(backbone, f"block{i}").blocks) for i in range(1, 4)] == [2, 4, 2]
    else:
        from torchvision.models.mobilenetv2 import InvertedResidual
        assert sum(isinstance(m, InvertedResidual) for m in backbone.modules()) == 17

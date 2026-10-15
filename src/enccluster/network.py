# Modified 2026-10-15: namespaced release integration; see THIRD_PARTY_LICENSES.
import torch
import torchmetrics
from enccluster.architectures import *

"Get model architecture to be used for experiment."
def get_backbone(name):
	if name=='resnet20':
		return ResNet20
	elif name=='convmixer256_8':
		return ConvMixer256_8
	elif name=='mobilenet_v2':
		return MobileNetV2
	elif name=='convnext':
		return ConvNeXT
	else:
		raise NotImplementedError(f'Architecture {name} not implemented')

class MobileNetV2(torch.nn.Module):
    """Official torchvision MobileNetV2, width multiplier 1.0, from scratch."""
    embeddings_dim = 1280

    def __init__(self):
        super().__init__()
        from torchvision.models import mobilenet_v2
        self.features = mobilenet_v2(weights=None).features
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.features(x).mean(dim=(2, 3)))

class CutMixCrossEntropyLoss(torch.nn.Module):

	def __init__(self, size_average=True):
		super().__init__()
		self.size_average = size_average

	@staticmethod
	def cross_entropy(input, target, size_average=True):
		logsoftmax = torch.nn.LogSoftmax(dim=1)
		if size_average:
			return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
		else:
			return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

	def forward(self, input, target):
		if len(target.size()) == 1:
			target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
			target = target.float().cuda()
		return __class__.cross_entropy(input, target, self.size_average)

class Model(torch.nn.Module):

	def __init__(self, model, num_classes=10):
		super(Model, self).__init__()
		self.num_classes = num_classes
		self.backbone = get_backbone(model)()
		self.classifier = torch.nn.Linear(self.backbone.embeddings_dim, self.num_classes)

	@property
	def dtype(self):
		return self.network.weight.dtype

	@property
	def device(self):
		return next(self.parameters()).device

	def forward(self, x):
		x = self.backbone(x)
		x = self.classifier(x)
		return x

	def compile(self, lr=1e-3):
		self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
		self.metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.device)

	def fit(self, ds, epochs=1, cutmix=False, verbose=False):
		self.train()
		score = []
		loss_fn = CutMixCrossEntropyLoss() if cutmix \
			else lambda p,y: torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(p, dim=1), y)
		for epoch in range(epochs):
			train_loss = 0.0
			self.metric.reset()
			for (x, y) in ds:
				x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
				self.optimizer.zero_grad()
				logits = self(x)
				_loss = loss_fn(logits,y)
				_loss.backward()
				self.optimizer.step()
				train_loss += _loss.item()
				y_preds = torch.argmax(logits, dim=1)
				self.metric(y_preds, torch.argmax(y, dim=1) if len(y.shape)>1 else y)
			train_loss /= len(ds)
			score.append(train_loss)
			acc = self.metric.compute()
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return {'loss': score, 'accuracy': acc}

	def evaluate(self, ds, verbose=False):
		self.metric.reset()
		self.eval()
		test_loss = 0.0
		with torch.no_grad():
			for (x, y) in ds:
				x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True).long()
				logits = self(x)
				y_prob = torch.nn.functional.log_softmax(logits, dim=1)
				test_loss += torch.nn.functional.nll_loss(y_prob, y).item()
				y_preds = torch.argmax(y_prob, dim=1)
				self.metric(y_preds, y)
		test_loss /= len(ds)
		acc = self.metric.compute()
		if verbose:
			print(f"Loss: {test_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return (test_loss, float(acc.detach().cpu().numpy()))

	def __str__(self):
		return self.__class__.__name__

	def get_params(self):
		return self.state_dict().items()

	def load_params(self, params):
		state = self.state_dict()
		state.update(params)
		self.load_state_dict(state, strict=False)

	def save_model(self, fp):
		if not fp.endswith('.pth'): fp += '.pth'
		torch.save(self.state_dict(), fp)

	@classmethod
	def load_model(cls, fp, model, num_classes=10):
		model = cls(model, num_classes)
		if not fp.endswith('.pth'): fp += '.pth'
		model.load_state_dict(torch.load(fp, map_location=lambda storage,_: storage))
		return model


'''
if __name__ == "__main__":
	#from torchinfo import summary
	#summary(ResNet20(), input_data=torch.rand(1,3,32,32))
	torch.autograd.set_detect_anomaly(True)
	model = ResNet20()
	pred = model(torch.randn(1, 3, 32, 32))
	y = torch.randn(pred.shape)
	torch.nn.MSELoss()(pred, y).backward()
'''
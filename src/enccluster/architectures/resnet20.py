import torch

class LambdaLayer(torch.nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)

class BasicBlock(torch.nn.Module):

	expansion = 1

	def __init__(self, in_planes, planes, stride=1, gn_init=None):
		super(BasicBlock, self).__init__()

		self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = torch.nn.GroupNorm(8, planes)
		__class__.gn_init(self.bn1)
		self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = torch.nn.GroupNorm(8, planes)
		__class__.gn_init(self.bn2)
		self.shortcut = torch.nn.Sequential()

		if stride != 1 or in_planes != planes:
			self.shortcut = LambdaLayer(lambda x: 
				torch.nn.functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

	@staticmethod
	def gn_init(m, zero_init=False):
		assert isinstance(m, torch.nn.GroupNorm)
		m.weight.data.fill_(0. if zero_init else 1.)
		m.bias.data.zero_()

	def forward(self, x):
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = torch.nn.functional.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out_relu = torch.nn.functional.relu(out)
		out_relu += self.shortcut(identity)
		return out

class ResNet20(torch.nn.Module):

	def __init__(self, block=BasicBlock, num_blocks=[3,3,3]):
		super(ResNet20, self).__init__()
		self.in_planes = 16
		self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = torch.nn.GroupNorm(8, 16)
		__class__.gn_init(self.bn1)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.embeddings_dim = 64

	@staticmethod
	def gn_init(m, zero_init=False):
		assert isinstance(m, torch.nn.GroupNorm)
		m.weight.data.fill_(0. if zero_init else 1.)
		m.bias.data.zero_()

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return torch.nn.Sequential(*layers)

	def forward(self, x):
		x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = torch.nn.functional.avg_pool2d(x, x.size()[3])
		x = x.view(x.size(0), -1)
		return x

'''
if __name__ == '__main__':
	import torchinfo # lazy import
	net = ResNet20()
	torchinfo.summary(net, input_data=torch.rand(1,3,32,32))
'''
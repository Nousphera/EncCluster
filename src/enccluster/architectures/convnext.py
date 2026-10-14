import torch

'''
class PreStem(torch.nn.Module):

	def __init__(self):
		super(PreStem, self).__init__()
		self.norm = torch.nn.BatchNorm2d(num_features=3, track_running_stats=False, affine=False)
		self.norm.running_mean = torch.tensor([0.4913997551666284 * 255, 0.48215855929893703 * 255, 0.4465309133731618 * 255])
		self.norm.running_var = torch.tensor([(0.24703225141799082 * 255) ** 2, (0.24348516474564 * 255) ** 2, (0.26158783926049628 * 255) ** 2])

	def forward(self, x):
		return self.norm(x)
'''

class LayerScale(torch.nn.Module):

	def __init__(self, init_values, projection_dim):
		super(LayerScale, self).__init__()
		self.init_values = init_values
		self.projection_dim = projection_dim
		self.gamma = torch.nn.Parameter(torch.ones(projection_dim) * init_values)

	def forward(self, x):
		return x * self.gamma

class Stem(torch.nn.Module):
	def __init__(self, num_filters):
		super(Stem, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=4)
		self.lnorm = torch.nn.LayerNorm(num_filters)

	def forward(self, x):
		x = self.conv(x)
		x = torch.nn.functional.pad(x, (1, 2, 1, 2)) # NOTE: Achieve 'same' padding
		return self.lnorm(x)

class DownSample(torch.nn.Module):

	def __init__(self, in_channels, out_channels, num_filters):
		super(DownSample, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=1)
		self.pool = torch.nn.MaxPool2d(kernel_size=2)
		self.lnorm = torch.nn.LayerNorm(out_channels)

	def forward(self, x):
		return self.lnorm(self.pool(self.conv(x)))

class Bottleneck(torch.nn.Module):

	def __init__(self, num_filters, factor=4, switch_init_a=1e-6, drop_rate=0.5, scale=1):
		super(Bottleneck, self).__init__()
		self.dconv = torch.nn.Conv2d(in_channels=num_filters, out_channels=int(num_filters*factor), kernel_size=3, padding='same', groups=num_filters)
		self.lnorm = self.lnorm = torch.nn.LayerNorm(num_filters//scale)
		self.dropout = torch.nn.Dropout(drop_rate)
		self.dense = torch.nn.Conv2d(in_channels=int(num_filters*factor), out_channels=num_filters, kernel_size=1)
		self.scale = LayerScale(switch_init_a, num_filters//scale)

	def forward(self, x):
		x = torch.nn.functional.gelu(self.lnorm(self.dconv(x)))
		x = self.dropout(x)
		return self.scale(self.dense(x))

class BasicBlock(torch.nn.Module):
	def __init__(self, num_filters, num_layers, factor=4, scale=1):
		super(BasicBlock, self).__init__()
		self.blocks = torch.nn.ModuleList([Bottleneck(num_filters, factor=factor, scale=scale) for i in range(num_layers)])

	def forward(self, input_tensor):
		x = input_tensor
		for block in self.blocks:
			x_sub = x
			x = block(x)
			x = x_sub + x  
		return x

class ConvNeXT(torch.nn.Module):

	def __init__(self):
		super(ConvNeXT, self).__init__()
		#self.prestem = PreStem()
		self.stem = Stem(32)
		self.block1 = BasicBlock(num_filters=32, num_layers=2, factor=2)
		self.downsample1 = DownSample(in_channels=32, out_channels=16, num_filters=64)
		self.block2 = BasicBlock(num_filters=64, num_layers=4, factor=4, scale=4)
		self.downsample2 = DownSample(in_channels=64, out_channels=8, num_filters=128)
		self.block3 = BasicBlock(num_filters=128, num_layers=2, factor=2, scale=16)
		self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
		self.lnorm = torch.nn.LayerNorm(128, elementwise_affine=True)
		self.embeddings_dim = 128

	def forward(self, x):
		x = self.stem(x)
		x = self.block1(x)
		x = self.downsample1(x)
		x = self.block2(x)
		x = self.downsample2(x)
		x = self.block3(x)
		x = self.pool(x)
		x = torch.flatten(x, 1)
		return torch.nn.functional.relu(self.lnorm(x))

'''
if __name__ == "__main__":
	import numpy as np
	from torchinfo import summary
	model = ConvNeXT()
	weights = np.concatenate([v.numpy().flatten() for v in model.state_dict().values()])
	print(np.sort(weights)[-20:])
	#summary(model, input_data=torch.rand(1,3,32,32))
'''

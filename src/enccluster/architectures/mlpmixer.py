import torch
from einops.layers.torch import Rearrange

class MLP1(torch.nn.Module):
	def __init__(self, num_patches, hidden_s, hidden_size, drop_p):
		super(MLP1, self).__init__()
		self.ln = torch.nn.LayerNorm(hidden_size)
		self.fc1 = torch.nn.Conv1d(num_patches, hidden_s, kernel_size=1)
		self.do1 = torch.nn.Dropout(p=drop_p)
		self.fc2 = torch.nn.Conv1d(hidden_s, num_patches, kernel_size=1)
		self.do2 = torch.nn.Dropout(p=drop_p)

	def forward(self, x):
		out = self.do1(torch.nn.functional.gelu(self.fc1(self.ln(x))))
		out = self.do2(self.fc2(out))
		return out+x

class MLP2(torch.nn.Module):

	def __init__(self, hidden_size, hidden_c, drop_p):
		super(MLP2, self).__init__()
		self.ln = torch.nn.LayerNorm(hidden_size)
		self.fc1 = torch.nn.Linear(hidden_size, hidden_c)
		self.do1 = torch.nn.Dropout(p=drop_p)
		self.fc2 = torch.nn.Linear(hidden_c, hidden_size)
		self.do2 = torch.nn.Dropout(p=drop_p)

	def forward(self, x):
		out = self.do1(torch.nn.functional.gelu(self.fc1(self.ln(x))))
		out = self.do2(self.fc2(out))
		return out+x

class MixerLayer(torch.nn.Module):

	def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p):
		super(MixerLayer, self).__init__()
		self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p)
		self.mlp2 = MLP2(hidden_size, hidden_c, drop_p)

	def forward(self, x):
		out = self.mlp1(x)
		out = self.mlp2(out)
		return out

class MLPMixer(torch.nn.Module):
	def __init__(self, in_channels=3,img_size=32, patch_size=4, hidden_size=128, hidden_s=512,
		hidden_c=64, num_layers=8, drop_p=0., is_cls_token=True):

		super(MLPMixer, self).__init__()
		num_patches = img_size // patch_size * img_size // patch_size
		self.is_cls_token = is_cls_token
		self.patch_emb = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size), Rearrange('b d h w -> b (h w) d'))
		if self.is_cls_token:
			self.cls_token = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
			num_patches += 1
		self.mixer_layers = torch.nn.Sequential(
			*[MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p)  for _ in range(num_layers)])
		self.ln = torch.nn.LayerNorm(hidden_size)
		self.embeddings_dim = hidden_size

	def forward(self, x):
		out = self.patch_emb(x)
		if self.is_cls_token:
			out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)
		out = self.mixer_layers(out)
		out = self.ln(out)
		out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
		return out

'''
if __name__ == '__main__':
	import torchinfo # lazy import
	net = MLPMixer()
	torchinfo.summary(net, input_data=torch.rand(1,3,32,32))
'''
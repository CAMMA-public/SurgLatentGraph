import torch
import sys

x = torch.load(sys.argv[1])
x['state_dict'] = {'detector.' + k: v for k, v in x['state_dict'].items()}
torch.save(x, sys.argv[1].replace('.pth', '_LG.pth'))

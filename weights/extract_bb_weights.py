import torch
import sys

x = torch.load(sys.argv[1])
x['state_dict'] = {k.replace('detector.backbone', 'backbone'): v for k, v in x['state_dict'].items() if 'backbone' in k}
torch.save(x, sys.argv[1].replace('.pth', '_bb.pth'))

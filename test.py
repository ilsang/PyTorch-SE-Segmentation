
import torch, os, glob, argparse
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader import CustomDset

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='data',
                        help='Path to images')
parser.add_argument('--weight_dir', type=str, default='weight',
                        help='path to pretrained weight')
parser.add_argument('--save_dir', type=str, default='weight',
                        help='Models are saved here')
parser.add_argument('--model', type=str, default='se_unet',
                        help='Select model, e.g., unet, se_unet, densenet, se_densenet')
parser.add_argument('--reduction_ratio', type=int, default=None,
                        help='Number of reduction ratio in SE block')
parser.add_argument('--growth_rate', type=int, default=16,
                        help='Number of growth_rate in Denseblock')
parser.add_argument('--gpu_ids', type=str, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_class', type=int, default=2,
                    help='Number of segmentation class')
parser.add_argument('--init_features', type=int, default=32, 
                    help='Initial feature map in the first conv')
parser.add_argument('--network_depth', type=int, default=4, 
                    help='Number of network depth')
parser.add_argument('--bottleneck', type=int, default=5, 
                    help='Number of bottleneck layer')
args = parser.parse_args()

print(args)

print(args)
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

testDataset = CustomDset(args.root_dir, 'test')
test_loader = DataLoader(testDataset, batch_size=args.batch_size,
                           shuffle=False, pin_memory=True)

print('Create model')

if args.model == 'unet':
    from model import UNet
    model = torch.nn.DataParallel(UNet(num_classes=args.num_class, init_features=args.init_features,
             network_depth=args.network_depth, bottleneck_layers=args.bottleneck,
                                       reduction_ratio = None), device_ids=gpu_ids)

if args.model == 'se_unet':
    from model import UNet
    model = torch.nn.DataParallel(UNet(num_classes=args.num_class, init_features=args.init_features,
             network_depth=args.network_depth, bottleneck_layers=args.bottleneck,
                                       reduction_ratio = args.reduction_ratio), device_ids=gpu_ids)

if args.model == 'densenet':
    from model import FCDenseNet
    model = torch.nn.DataParallel(FCDenseNet(in_channels=args.num_class, down_blocks=(4,5,7,10,12), 
                                             up_blocks=(12,10,7,5,4), bottleneck_layers=args.bottleneck, 
                                             growth_rate=args.growth_rate,
                                             out_chans_first_conv=args.init_features,
                                             n_classes=args.num_class),
                                  device_ids=gpu_ids)
    
if args.model == 'se_densenet':
    from model import FCDenseNet
    model = torch.nn.DataParallel(FCDenseNet(reduction_ratio = args.reduction_ratio, 
                                             in_channels=args.num_class, down_blocks=(4,5,7,10,12), 
                                             up_blocks=(12,10,7,5,4), bottleneck_layers=args.bottleneck, 
                                             growth_rate=args.growth_rate,
                                             out_chans_first_conv=args.init_features, 
                                             n_classes=args.num_class),
                                  device_ids=gpu_ids)

print('Load weight from %s' % args.weight_dir)
model.load_state_dict(torch.load(args.weight_dir))
model.cuda(gpu_ids[0])
print(model)
print('Start inference')

for step, (patient, adc, b1000, mask) in enumerate(test_loader):
    adc = adc.cuda(gpu_ids[0], async=True)
    b1000 = b1000.cuda(gpu_ids[0], async=True)
    
    inputs = Variable(torch.cat([adc,b1000], 1), volatile=True)
    targets = Variable(mask.long().cuda(gpu_ids[0], async=True), volatile=True)

    outputs = model(inputs)
    outputs = outputs.max(1)[1].cpu().data.numpy()
    
    filename = os.path.join(args.save_dir, patient[0] + '.npy')
    np.save(filename, outputs)

print('finish')
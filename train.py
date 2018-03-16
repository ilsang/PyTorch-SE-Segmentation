
import torch, os, glob, tqdm, argparse
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from optim import CrossEntropyLoss2d, adjust_learning_rate
from dataloader import CustomDset
from visualization import Dashboard, gray2rgb, gray2rgb_norm
from metric import dice_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='data',
                        help='Path to images')
parser.add_argument('--save_dir', type=str, default='weight',
                        help='Models are saved here')
parser.add_argument('--weight_dir', type=str, default=None,
                        help='path to pretrained weight')
parser.add_argument('--model', type=str, default='unet',
                        help='Select model, e.g., unet, se_unet, densenet, se_densenet')
parser.add_argument('--reduction_ratio', type=int, default=None,
                        help='Number of reduction ratio in SE block')
parser.add_argument('--growth_rate', type=int, default=16,
                        help='Number of growth_rate in Denseblock')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu_ids', type=str, default=0,
                       help='gpu ids: e.g. 0  0,1,2, 0,2')
parser.add_argument('--lr', type=float, default=0.0001, 
                    help='Initial learning rate')
parser.add_argument('--num_class', type=int, default=2,
                    help='Number of segmentation class')
parser.add_argument('--num_epochs', type=int, default=200, 
                    help='Number of epoch')
parser.add_argument('--step_print', type=int, default=100, 
                    help='Frequency of showing training results')
parser.add_argument('--epochs_save', type=int, default=5, 
                    help='Frequency of saving weight at the end of epochs')
parser.add_argument('--init_features', type=int, default=32, 
                    help='Initial feature map in the first conv')
parser.add_argument('--network_depth', type=int, default=4, 
                    help='Number of network depth')
parser.add_argument('--bottleneck', type=int, default=5, 
                    help='Number of bottleneck layer')
parser.add_argument('--down_blocks', type=str, default=None, 
                    help='Number of bottleneck layer')
parser.add_argument('--up_blocks', type=str, default=None, 
                    help='Number of bottleneck layer')
parser.add_argument('--port', type=int, default=8097, 
                    help='Visdom port of the web display')
parser.add_argument('--visdom_env_name', type=str, default='SE_segmentation',
                        help='Name of current environment in visdom')
args = parser.parse_args()


print(args)
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

if args.down_blocks is not None:
    blocks = args.down_blocks.split(',')
    down_blocks = []
    for block in blocks:
        id = int(str_id)
        if id >= 0:
            down_blocks.append(id)

if args.up_blocks is not None:
    blocks = args.up_blocks.split(',')
    up_blocks = []
    for block in blocks:
        id = int(str_id)
        if id >= 0:
            up_blocks.append(id)

torch.cuda.set_device(gpu_ids[0])
board = Dashboard(args.port, args.visdom_env_name)

trainDataset = CustomDset(args.root_dir,'train')
train_loader = DataLoader(trainDataset, batch_size=args.batch_size,
                           shuffle=True, pin_memory=True)
valDataset = CustomDset(args.root_dir,'val')
val_loader = DataLoader(valDataset, batch_size=args.batch_size,
                           shuffle=True, pin_memory=True)

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
    model = torch.nn.DataParallel(FCDenseNet(in_channels=args.num_class, down_blocks=down_blocks, 
                                             up_blocks=up_blocks, bottleneck_layers=args.bottleneck, 
                                             growth_rate=args.growth_rate,
                                             out_chans_first_conv=args.init_features,
                                             n_classes=args.num_class),
                                  device_ids=gpu_ids)
    
if args.model == 'se_densenet':
    from model import FCDenseNet
    model = torch.nn.DataParallel(FCDenseNet(reduction_ratio = args.reduction_ratio, 
                                             in_channels=args.num_class, down_blocks=down_blocks, 
                                             up_blocks=up_blocks, bottleneck_layers=args.bottleneck, 
                                             growth_rate=args.growth_rate,
                                             out_chans_first_conv=args.init_features, 
                                             n_classes=args.num_class),
                                  device_ids=gpu_ids)

if args.weight_dir is not None:
    print('Load weight from %s' % args.weight_dir)
    model.load_state_dict(torch.load(args.weight_dir))

model.cuda(gpu_ids[0])
print(model)

weight = torch.ones(args.num_class)
criterion = CrossEntropyLoss2d(weight.cuda())
optimizer = Adam(model.parameters(), lr=args.lr)

print('Start training')

for epoch in range(1, args.num_epochs+1):
    adjust_learning_rate(optimizer, args.lr,epoch)
    
    model.train()
    
    epoch_train_loss = []

    for step, (patient, adc, b1000, mask) in enumerate(train_loader):
        adc = adc.cuda(gpu_ids[0], async=True)
        b1000 = b1000.cuda(gpu_ids[0], async=True)
        
        images = torch.cat([adc,b1000], 1)
        targets = mask.long().cuda(gpu_ids[0], async=True)

        inputs = Variable(images, requires_grad=True)
        targets = Variable(targets)
        outputs = model(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets[:, 0])
        loss.backward()
        optimizer.step()

        epoch_train_loss.append(loss.data[0])
        step_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        if step % args.step_print == 0:
            print('train loss: %.7f (step: %d)' % (step_train_loss, step))
    
        
    train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
    board.plot("loss", "train", epoch+1, train_loss)

    
    if epoch % args.epochs_save == 0:
        filename = os.path.join(args.save_dir, 'model-%d.pth' % epoch)
        torch.save(model.state_dict(), filename)
        print('save: %s (epoch: %d)' % \
             (filename, epoch))

    model.eval()
    
    epoch_val_loss = []
    performances = []
    
    for step, (patient, adc, b1000, mask) in enumerate(val_loader):
        adc = adc.cuda(gpu_ids[0], async=True)
        b1000 = b1000.cuda(gpu_ids[0], async=True)
        
        images = torch.cat([adc,b1000], 1)
        labels = mask.long().cuda(gpu_ids[0], async=True)

        
        inputs = Variable(images, volatile=True)
        targets = Variable(labels, volatile=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets[:, 0])
        
        performance = dice_tensor(outputs.max(1)[1], targets[:,0])
        epoch_val_loss.append(loss.data[0])
        performances.append(performance.data[0])
        
        _adc = inputs[0][0].unsqueeze(0).cpu().data
        _b1000 = inputs[0][1].unsqueeze(0).cpu().data

    board.image(gray2rgb_norm(_adc),
        'patient : %s, ADC (epoch: %d)' % (patient[0], epoch))
    board.image(gray2rgb(_b1000),
        'patient : %s, b1000 (epoch: %d)' % (patient[0], epoch))

    board.image(outputs[0].max(0)[1].cpu().data.double(),
        'patient : %s, output (epoch: %d)' % (patient[0], epoch))
    board.image(gray2rgb(targets[0].cpu().data.double()),
        'patient : %s, target (epoch: %d)' % (patient[0], epoch))
        
    val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
    print('val loss: %.7f, epoch: %d' % (val_loss, epoch))
    board.plot("loss", "val", epoch, val_loss)
    val_measure = sum(performances) / len(performances)
    print('val Dice: %.7f, epoch: %d' % (val_measure, epoch))
    board.plot("Dice", "val", epoch, val_measure)
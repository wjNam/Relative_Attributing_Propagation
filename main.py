import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from glob import glob
import imageio
import torch.backends.cudnn as cudnn
from modules.vgg import vgg16, vgg16_bn
from modules.resnet import resnet50, resnet101
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import argparse
def enlarge_image(img, scaling = 3):
    if scaling < 1 or not isinstance(scaling,int):
        print ('scaling factor needs to be an int >= 1')

    if len(img.shape) == 2:
        H,W = img.shape
        out = np.zeros((scaling*H, scaling*W))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling] = img[h,w]
    elif len(img.shape) == 3:
        H,W,D = img.shape
        out = np.zeros((scaling*H, scaling*W,D))
        for h in range(H):
            fh = scaling*h
            for w in range(W):
                fw = scaling*w
                out[fh:fh+scaling, fw:fw+scaling,:] = img[h,w,:]
    return out
def hm_to_rgb(R, scaling = 3, cmap = 'bwr', normalize = True):
    cmap = eval('matplotlib.cm.{}'.format(cmap))
    if normalize:
        R = R / np.max(np.abs(R)) # normalize to [-1,1] wrt to max relevance magnitude
        R = (R + 1.)/2. # shift/normalize to [0,1] for color mapping
    R = R
    R = enlarge_image(R, scaling)
    rgb = cmap(R.flatten())[...,0:3].reshape([R.shape[0],R.shape[1],3])
    return rgb
def visualize(relevances, img_name):
    # visualize the relevance
    n = len(relevances)
    heatmap = np.sum(relevances.reshape([n, 224, 224, 1]), axis=3)
    heatmaps = []
    for h, heat in enumerate(heatmap):
        maps = hm_to_rgb(heat, scaling=3, cmap = 'seismic')
        heatmaps.append(maps)
        imageio.imsave('./results/'+ method + '/' + data_name + img_name, maps,vmax=1,vmin=-1)
def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    print('Pred cls : '+str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = Variable(T).cuda()
    return Tt
cudnn.benchmark = True
# Args
parser = argparse.ArgumentParser(description='Interpreting the decision of classifier')
parser.add_argument('--method', type=str, default='RAP', metavar='N',
                    help='Method : LRP or RAP')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture: vgg / resnet')
args = parser.parse_args()
num_workers = 0
batch_size = 1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_name = 'imagenet/'

# define data loader

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('./data/'+ data_name,
                          transforms.Compose([
                              transforms.Scale([224, 224]),
                              transforms.ToTensor(),
                              normalize,
                          ])),
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
if args.arc == 'vgg':
    model = vgg16(pretrained=True).cuda()
elif args.arc == 'resnet':
    # model = resnet101(pretrained=True).cuda()
    model = resnet50(pretrained=True).cuda()
# method = LRP or RAP
method = args.method
model.eval()
for idx, (input, label) in enumerate(val_loader):
    input = Variable(input, volatile=True).cuda()
    input.requires_grad = True
    img_name = val_loader.dataset.imgs[idx][0].split('\\')[1]
    output = model(input)
    T = compute_pred(output)
    if method == 'LRP':
        Res = model.relprop(R = output * T, alpha= 1).sum(dim=1, keepdim=True)
    else:
        RAP = model.RAP_relprop(R=T)
        Res = (RAP).sum(dim=1, keepdim=True)
    # Check relevance value preserved
    print('Pred logit : ' + str((output * T).sum().data.cpu().numpy()))
    print('Relevance Sum : ' + str(Res.sum().data.cpu().numpy()))
    # save results
    heatmap = Res.permute(0, 2, 3, 1).data.cpu().numpy()
    visualize(heatmap.reshape([batch_size, 224, 224, 1]), img_name)
print('Done')
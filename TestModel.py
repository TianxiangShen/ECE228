from __future__ import print_function
import sys
from PIL import Image
import os.path
from Parameters import Params
from torchvision import transforms
from torchvision import utils as tv
from Dataset import ImageSet
import torch
from torch.utils.data import DataLoader
from network_elements import Mapping
from torch.autograd import Variable

args = sys.argv;
path = args[-1];
model = args[-2];

p = Params();
p.model = model;
test_transforms = [];

if "-r" in args[1:-2]:
    p.dir = 1
if "-o" in args[1:-2]:
    outputInd = args.index("-o");
    p.outputPath = args[outputInd+1];
    if not os.path.exists(p.outputPath):
	os.makedirs(p.outputPath);
else:
    p.outputPath = "./outputs/"+model+"/";
    if not os.path.exists(p.outputPath):
        os.makedirs(p.outputPath);
    if (p.dir):
        tempPath = path.split('/')
        print(tempPath)
        if (path[-1] == '/'):
            p.outputPath += tempPath[-2] + "/";
        else:
            p.outputPath += tempPath[-1] + "/";
        if not os.path.exists(p.outputPath):
            os.mkdir(p.outputPath);
                
if "-d" in args[1:-2]:
    outputInd = args.index("-d");
    p.direction = args[outputInd+1];
else:
	p.direction = "yx";
    
if "-s" in args[1:-2]:
    outputInd = args.index("-s");
    p.scale = float(args[outputInd+1]);
else:
    p.scale = 1.0;
    
        
print("Model under test is " + p.model);
print("The output path is "+p.outputPath);

images = ImageSet();
test_transforms += [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))];
if (p.dir):
    if os.path.exists(path):
        print("Processing set in directory: "+path);
        set
    else:
        print("No directory exists at path: "+path, file=sys.stderr); 
    images.loadImageSet(path, test_transforms, "test", p.scale);
    
else:
    if os.path.exists(path):
        print("Processing file at path: " + path);
    else:
        print("No file exists at path: "+path, file=sys.stderr);
    images.loadImage(path, test_transforms, "test", p.scale);

imgLoader = DataLoader(images, 1, shuffle=False);

if (p.direction == 'xy'):
	modelfile = torch.load("./model/"+p.model+"/G.data");
else:
	modelfile = torch.load("./model/"+p.model+"/F.data");

m19 = modelfile['model.19.weight'];
im_size = m19.size(0);
F = Mapping(3,3,im_size);
F.cuda();
F.load_state_dict(modelfile);
F.eval();
    
for i, img in enumerate(imgLoader):
    imgsize = img['img'].size();
    imname = img['path']
    print(imname)
    img_gpu = torch.cuda.FloatTensor(1,3,imgsize[2],imgsize[3]);
    img_var = Variable(img_gpu.copy_(img['img']))
    result = 0.5*(F(img_var).data+1.0);
    tv.save_image(result, p.outputPath+'out_' + imname[0]);

import torch.nn as tnn
import torch
from torch.autograd import Variable
import random


#Define residual block
#Architecture is:
    # Reflection padding
    # 3x3 Convolution-InstanceNorm-ReLU
        # 128 filters
    # Reflection padding
    # 3x3 Convolution-InstanceNorm
        # 128 filters
class ResBlock(tnn.Module):
    def __init__(self, size):
        super(ResBlock, self).__init__()
        
        res_model = [tnn.ReflectionPad2d(1),
                     tnn.Conv2d(size, size, 3),
                     tnn.InstanceNorm2d(size),
                     tnn.ReLU(True),
                     tnn.ReflectionPad2d(1),
                     tnn.Conv2d(size, size, 3),
                     tnn.InstanceNorm2d(size)]
        
        self.model = tnn.Sequential(*res_model)
        
    def forward(self, x):
        return x + self.model(x)

#Define mapping G: X->Y
# M = number of residual blocks

#Architecture is:
# Reflection padding
# 7x7 Convolution-InstanceNorm-ReLU
    # 32 filters, stride 1
# 3x3 Convolution-InstanceNorm-RelU
    # 64 filters, stride 2
# 3x3 Convolution-InstanceNorm-ReLU
    # 128 filters, stride 2
# M x Residual Block
# 3x3 Fractional-Strided-Convolution-InstanceNorm-ReLU
    # 64 filters, stride 1/2
# 3x3 Fractional-Strided-Convolution-InstanceNorm-ReLU
    # 32 filters, stride 1/2
# 7x7 Convolution-InstanceNorm-ReLU
    # 3 filters, stride 1

class Mapping(tnn.Module):
    
    #Constructor
    def __init__(self, im_in_channels, im_out_channels, img_dim, num_filt=64):
        super(Mapping, self).__init__()
        
        #Assign the number of residual blocks
        num_res_blocks = 0
        if (img_dim == 128):
            num_res_blocks = 6
        elif (img_dim >= 256):
            num_res_blocks = 9
        
        gen_model = [tnn.ReflectionPad2d(3),
                     tnn.Conv2d(im_in_channels, num_filt, 7),
                     tnn.InstanceNorm2d(num_filt),
                     tnn.ReLU(True)]
        
        #Downsample
        in_feat = 64
        out_feat = in_feat*2;
        for i in range(2):
            gen_model += [tnn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1),
                          tnn.InstanceNorm2d(out_feat),
                          tnn.ReLU(True)]
            in_feat = out_feat
            out_feat = in_feat*2
            
        for i in range(num_res_blocks):
            gen_model += [ResBlock(in_feat)]
            
        out_feat = in_feat//2
        for i in range(2):
            gen_model += [tnn.ConvTranspose2d(in_feat, out_feat, 3, stride = 2,
                                              padding = 1, output_padding = 1),
                          tnn.InstanceNorm2d(out_feat),
                          tnn.ReLU(True)]
            in_feat = out_feat
            out_feat = in_feat//2
                        
        gen_model += [tnn.ReflectionPad2d(3),
                  tnn.Conv2d(num_filt, im_out_channels, 7),
                  tnn.Tanh()]
        
        self.model = tnn.Sequential(*gen_model)
        
    #Forward propagation
    def forward(self, x):
        return self.model(x)
        
    
#Discriminator different than CycleGAN model
class Discriminator(tnn.Module):
    def __init__(self, im_in_channels):
        super(Discriminator, self).__init__()
        
        model = [   tnn.Conv2d(im_in_channels, 64, 4, stride=2, padding=1),
                    tnn.LeakyReLU(0.2, inplace=True) ]

        model += [  tnn.Conv2d(64, 128, 4, stride=2, padding=1),
                    tnn.InstanceNorm2d(128), 
                    tnn.LeakyReLU(0.2, inplace=True) ]

        model += [  tnn.Conv2d(128, 256, 4, stride=2, padding=1),
                    tnn.InstanceNorm2d(256), 
                    tnn.LeakyReLU(0.2, inplace=True) ]

        model += [  tnn.Conv2d(256, 512, 4, padding=1),
                    tnn.InstanceNorm2d(512), 
                    tnn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [tnn.Conv2d(512, 1, 4, padding=1)]
        self.model = tnn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return tnn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)



def conv_norm_weights(m):
    if type(m) == tnn.Conv2d:
        tnn.init.normal(m.weight.data, 0.0, 0.02)
        
class LRDecay():
    def __init__(self, decayStart, decayEnd):
        self.decayStart = decayStart
        self.decayEnd = decayEnd
        
    def step(self, epoch):
        return 1.0-(epoch >= self.decayStart)*((epoch-self.decayStart)/(self.decayEnd-self.decayStart))
        
class MapBuffer():
    def __init__(self):
        self.size = 50
        self.data = []
        
    def cycle(self, data):

        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

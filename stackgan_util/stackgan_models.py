import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

###ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.conv_block(x)
    
def upsamp_conv(in_dim, out_dim, upscale_factor):
        return nn.Sequential(
            nn.PixelShuffle(upscale_factor = upscale_factor),
            nn.Conv2d(in_dim//(upscale_factor**2), out_dim, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.1)
        )
    
class Generator_Layer1(nn.Module):
    """
    Input shape: (N, in_dim)
    Output shape: (N, 3, 64, 64)
    """
    def __init__(self, in_dim, tag_dim1, tag_dim2, dim=16):
        super(Generator_Layer1, self).__init__()
        
        ######################################
        self.t1 = nn.Linear(tag_dim1, 128, bias=True)  #hair
        self.t2 = nn.Linear(tag_dim2, 128, bias=True)  #eyes
        ######################################
        self.l1 = nn.Sequential(
            nn.Linear(in_dim+256, dim * 64 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 64 * 4 * 4),
            nn.ReLU()
        )
        self.rdb1 = ResidualBlock(dim * 64)
        
        self.l2_5 = nn.Sequential(
            upsamp_conv(dim * 64, dim *16, 2),                                            #(dim * 16, 8, 8)
            upsamp_conv(dim * 16, dim *4, 2),                                            #(dim * 4, 16, 16)
            ResidualBlock(dim * 4),
            nn.Conv2d(dim *4, dim *16, 3, stride=1, padding=1, padding_mode='reflect'),  #(dim * 16, 16, 16)
            nn.BatchNorm2d(dim *16),
            nn.LeakyReLU(0.1),
            upsamp_conv(dim * 16, dim * 4, 2),                                           #(dim * 4, 32, 32)
            upsamp_conv(dim * 4, dim , 2),                                               #(dim, 64, 64)
            #nn.Conv2d(dim , 3, 3, stride=1, padding=1, padding_mode='reflect'),          #(3, 64, 64)
            #nn.Tanh()
        )
        self.apply(weights_init)

    #def forward(self, x, tag_h, tag_e, tag_p):  #patch
    def forward(self, x, tag_h, tag_e):
        embed1 = self.t1(tag_h)
        #######################################################
        embed2 = self.t2(tag_e)
        #######################################################
        y = torch.cat((x, embed1, embed2), -1)
        y = self.l1(y)                     #(dim * 64 * 4 * 4,)
        y = y.view(y.size(0), -1, 4, 4)    #(dim * 64, 4, 4)
        y = self.rdb1(y)                   #(dim * 64, 4, 4)
        y = self.l2_5(y)                   #(dim, 64, 64)
        return y
    

class Generator_Layer2(nn.Module):
    
    def __init__(self, in_dim, tag_dim1, tag_dim2, dim=16):
        super(Generator_Layer2, self).__init__()
        
        ######################################
        self.t1 = nn.Linear(tag_dim1, 128, bias=True)  #hair
        self.t2 = nn.Linear(tag_dim2, 128, bias=True)  #eyes
        ######################################
        self.rdb1 = ResidualBlock(in_dim+256)
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim+256, dim *16, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(dim *16),
            nn.LeakyReLU(0.1),
            upsamp_conv(dim * 16, dim * 4, 2),
            ResidualBlock(dim * 4),
            ResidualBlock(dim * 4),
            upsamp_conv(dim * 4, dim , 2),
        )
        self.apply(weights_init)
        
    def forward(self, x, tag_h, tag_e):
        embed1 = self.t1(tag_h)
        embed1 = embed1.reshape(embed1.size(0), -1, 1, 1)   #(, 128, 1, 1)
        embed1 = embed1.tile((1, 1, 64, 64))                  #(, 128, 64, 64)
        #######################################################
        embed2 = self.t2(tag_e)
        embed2 = embed2.reshape(embed2.size(0), -1, 1, 1)
        embed2 = embed2.tile((1, 1, 64, 64))                  #(, 128, 64, 64)
        #######################################################
        y = torch.cat((x, embed1, embed2), dim=1)
        y = self.rdb1(y)
        y = self.l1(y)
        return y
    
class Generator_Image(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        
        self.img = nn.Sequential(
            nn.Conv2d(in_dim , 3, 3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        self.apply(weights_init)
    def forward(self, x):
        y = self.img(x)
        return y

    

def conv_bn_lrelu(in_dim, out_dim, layersize):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        #nn.BatchNorm2d(out_dim),
        nn.LayerNorm([out_dim, layersize, layersize]),
        nn.LeakyReLU(0.2),
    )
    
    
class Discriminator_Layer1(nn.Module):
    """
    Input shape: (N, 3, 64, 64)
    Output shape: (N, )
    """
    def __init__(self, in_dim, tag_dim1, tag_dim2, dim=64):
        super(Discriminator_Layer1, self).__init__()
        
        ################################################
        self.t1 = nn.Linear(tag_dim1, 128, bias=True)  #hair
        self.t2 = nn.Linear(tag_dim2, 128, bias=True)  #eyes
        ################################################
            
        """ Medium: Remove the last sigmoid layer for WGAN. """
        
        self.inputsize=64   #3*64*64
        
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), #64*32*32
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 4, layersize=self.inputsize//4),     #256*16*16
            conv_bn_lrelu(dim * 4, dim * 8, layersize=self.inputsize//8), #512*8*8

        )
        #####realistic#############
        self.rloss = nn.Sequential(
            conv_bn_lrelu(dim * 8, dim * 16, layersize=self.inputsize//16), #1024*4*4
            nn.Conv2d(dim * 16, dim * 8, 3, 1, 1),   #512*4*4
            nn.Conv2d(dim * 8, dim * 4, 4),          #256*1*1
            nn.Flatten(),                     #256
            nn.Linear(dim * 4, 1, bias=True),  #1
            nn.Sigmoid(), 
        )
        #####conditional(text matching)######
        self.closs = nn.Sequential(
        #    nn.Conv2d(dim * 16+256, dim * 8, 3, 1, 1),
            conv_bn_lrelu(dim * 8+256, dim * 16, layersize=self.inputsize//16),
            nn.Conv2d(dim * 16, dim * 8, 3, 1, 1),
            nn.Conv2d(dim * 8, dim * 4, 4),
            nn.Flatten(), 
            nn.Linear(dim * 4, 1, bias=True), 
            nn.Sigmoid(), 
        )
        #######################################
        self.apply(weights_init)
        
    #def forward(self, x, tag_h, tag_e, tag_p):    #patch
    def forward(self, x, tag_h, tag_e):
        embed1 = self.t1(tag_h)                     #(, 128)
        embed1 = embed1.reshape(embed1.size(0), -1, 1, 1)   #(, 128, 1, 1)
        embed1 = embed1.tile((1, 1, 8, 8))                  #(, 128, 8, 8)
        ########################################################################
        embed2 = self.t2(tag_e)
        ########################################################################
        embed2 = embed2.reshape(embed2.size(0), -1, 1, 1)
        embed2 = embed2.tile((1, 1, 8, 8))                  #(, 128, 8, 8)
        y = self.ls(x)                                      #(, 512, 8, 8)
        y_realistic = self.rloss(y) 
        y_conditional = torch.cat((y, embed1, embed2), dim=1)           #(, 512+256, 8, 8)
        y_conditional = self.closs(y_conditional)
        #y = y.view(-1)
        return y_realistic, y_conditional
    
    
class Discriminator_Layer2(nn.Module):
    """
    Input shape: (N, 3, 256, 256)
    Output shape: (N, 1, 2, 2)
    """
    def __init__(self, in_dim, tag_dim1, tag_dim2, dim=64):
        super(Discriminator_Layer2, self).__init__()
        
        ################################################
        self.t1 = nn.Linear(tag_dim1, 128, bias=True)  #hair
        self.t2 = nn.Linear(tag_dim2, 128, bias=True)  #eyes
        ################################################
            
        """ Medium: Remove the last sigmoid layer for WGAN. """
        
        self.inputsize=256   #3*256*256
        
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), #64*128*128
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 4, layersize=self.inputsize//4),     #256*64*64

        )
        #####realistic#############
        self.rloss = nn.Sequential(
            conv_bn_lrelu(dim * 4, dim * 8, layersize=self.inputsize//8), #512*32*32
            nn.Conv2d(dim * 8, dim * 8, 5, 4, 2),   #512*8*8
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 8, dim, 5, 4, 2),          #64*2*2
            nn.Conv2d(dim , 1, 3, 1, 1),               #1*2*2
            nn.Sigmoid(), 
        )
        #####conditional(text matching)######
        self.closs = nn.Sequential(
        #    nn.Conv2d(dim * 16+256, dim * 8, 3, 1, 1),
            nn.Conv2d(dim * 4+256, dim * 8, 5, 4, 2),                      #512*16*16
            nn.LayerNorm([dim * 8, self.inputsize//16, self.inputsize//16]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim * 8, dim * 8, 5, 4, 2),                          #512*4*4
            nn.Conv2d(dim * 8, dim * 4, 5, 4, 2),                          #256*1*1
            nn.Flatten(), 
            nn.Linear(dim * 4, 1, bias=True), 
            nn.Sigmoid(), 
        )
        #######################################
        self.apply(weights_init)
        
    def forward(self, x, tag_h, tag_e):
        embed1 = self.t1(tag_h)                     #(, 128)
        embed1 = embed1.reshape(embed1.size(0), -1, 1, 1)   #(, 128, 1, 1)
        embed1 = embed1.tile((1, 1, 64, 64))                  #(, 128, 64, 64)
        ########################################################################
        embed2 = self.t2(tag_e)
        ########################################################################
        embed2 = embed2.reshape(embed2.size(0), -1, 1, 1)
        embed2 = embed2.tile((1, 1, 64, 64))                  #(, 128, 64, 64)
        y = self.ls(x)                                      #(, 256, 64, 64)
        y_realistic = self.rloss(y)                         #(, 1, 2, 2)
        #y_realistic = y_realistic.view(-1, 2, 2)            #(, 2, 2)
        y_realistic = y_realistic.view(-1, 4)                #(, 4)
        y_realistic = torch.mean(y_realistic, 1, True)       #(, 1)
        y_conditional = torch.cat((y, embed1, embed2), dim=1)           #(, 256+256, 64, 64)
        y_conditional = self.closs(y_conditional)                       #(, 1)
        
        return y_realistic, y_conditional
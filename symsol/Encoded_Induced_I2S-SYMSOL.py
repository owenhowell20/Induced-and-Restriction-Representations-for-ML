#!/usr/bin/env python
# coding: utf-8

# # Image to Sphere Via Induced Represenations For Pose Estimation on PASCAL# 

import wandb
import random
import datetime

### import relevent packages
import torch
import sys
import numpy as np
from e2cnn import gspaces
from e2cnn import nn
from e2cnn import group
import torchvision
from e3nn import o3
import e3nn
import represenations_opps as rep_ops
import healpy as hp


import time
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

###import pickle5 as pickle
import pickle

### check cuda read
torch.cuda.is_available()

### use gpus if avalible, otherwise use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### number of gpus avalible
num_gpu = torch.cuda.device_count() 

print("Number of GPUs avalible:", num_gpu )

###set parameters
learning_rate = 0.000002
lmax = 5
kmax = 10
f_out = 250
train_batch_size = 5

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Induced_Rep",
    name=str( datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") ),

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "Induced",
    "dataset": "SYMSOL",
    "lmax": lmax ,
    "kmax": kmax,
    "f_out": f_out,
    "batch_size": train_batch_size,
    "mulplicities_SO3": "sphere" ,
	}
)



### defining a SO2 convolution layer
SO2_act = gspaces.Rot2dOnR2(N=-1,maximum_frequency=kmax)


### convert SO2 reps to e2cnn format
### This should be put in seperate script
def convert_SO2(  input_rep_dict ):
    total_rep = []
    for k in input_rep_dict.keys():
        mulplicites = input_rep_dict[k]        
        total_rep = total_rep + mulplicites * [SO2_act.irrep(int(k))]

    return total_rep


### SWIN Image encoder
class ImageEncoder(torch.nn.Module):
    '''Define an image encoding network to process image into dense feature map

    Any standard convolutional network or vision transformer could be used here. 
    In the paper, we use ResNet50 pretrained on ImageNet1K for a fair comparison to
    the baselines.  Here, we show an example using a pretrained SWIN Transformer.

    When using a model from torchvision, make sure to remove the head so the output
    is a feature map, not a feature vector
    '''
    def __init__(self):
        super().__init__()
        self.layers = torchvision.models.swin_v2_t(weights="DEFAULT")

        # last three modules in swin are avgpool,flatten,linear so change to Identity
        self.layers.avgpool = torch.nn.Identity()
        self.layers.flatten = torch.nn.Identity()
        self.layers.head = torch.nn.Identity()

        # we will need shape of feature map for later
        dummy_input = torch.zeros((1, 3, 224, 224))
        self.output_shape = self(dummy_input).shape[1:]

    def forward(self, x):
        return self.layers(x)




### Different approch, done using just matrix multiplications
### This is also way too slow
### defining an induction layer from SO(2) to SO(3)
class Induction_Layer_III( torch.nn.Module ):
    
    ''' The Induction Layer is a Linear Layer that takes an SO2 represetation and outputs SO3 representations
    For more details on the induction layer, please read the attached notes!
    
    Class Induction_Layer teturns matrix valued coefficients of spherical harmonics
    :channels: Number of channels in image
    :image_shape: integer, images must be square!!!
    : kmax: maximum degree of so2 harmonics
    :lmax: maximum degree of so3 harmonics
    : rep_in  : input SO2 representation as dict
    : rep_out : output SO3 representation as dict
    
    '''
    def __init__(self, channels:int , image_shape:int , k_max:int , L_max: int, dict_rep_in :dict , dict_rep_out:dict ):
        
        super().__init__()
        self.k_max = k_max
        self.lmax = L_max
        self.channels = channels
        self.image_shape = image_shape
        
        ### tensor product represenations as list
        self.tensor_reps = []
        
        ### input and output representations as dict
        self.dict_rep_in = dict_rep_in
        self.dict_rep_out = dict_rep_out
        
        ### defining SO2 action
        SO2_act = gspaces.Rot2dOnR2(N=-1,maximum_frequency=self.k_max)
        
        ### compute restriction of SO(3) output SO(2) representation
        restrict = rep_ops.compute_restriction_SO3( self.dict_rep_out )
        self.rep_out = convert_SO2(  restrict  )
        
        ### output feature types
        self.feat_type_out = nn.FieldType( SO2_act, self.rep_out  )
        
        
        ### compute direct sum representation
        total_rep = []
        for l in range( 0 , L_max ):
            
            tensor_rep_in = rep_ops.compute_tensor_SO2_l_fold( self.dict_rep_in , l )
            
            rep_in = convert_SO2( tensor_rep_in )

            
            total_rep = total_rep + rep_in
            
        feat_type_in = nn.FieldType( SO2_act, total_rep  )
            
        self.conv = nn.R2Conv( feat_type_in , self.feat_type_out , kernel_size=self.image_shape)        
        
        
            
    def forward(self, x):
                
        F = self.conv.expand_parameters()[0]
        F_val_cat = torch.split( F , split_size_or_sections= self.channels , dim=1)
        g_split = torch.stack( F_val_cat ,dim=0 )
            
        ###now contract
        y = torch.einsum('ijklm , aklm -> aji' ,  g_split , x )
        
        return y



### Ask david if there is max beta that is optimal
def s2_healpix_grid(rec_level: int=0, max_beta: float=np.pi/6):
    """Returns healpix grid up to a max_beta
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    m = hp.query_disc(nside=n_side, vec=(0,0,1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()

def flat_wigner(lmax, alpha, beta, gamma):
    return torch.cat([ (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1) ], dim=-1)

### this should be changed,
### or just set output to be equal to hidden so3 layer
def s2_irreps(lmax):
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

def so3_irreps(lmax):
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])



### defining convolution over sphere
### just make sure that input so3 rep matches output so3 rep
class S2Conv(torch.nn.Module):

    def __init__(self, f_in: int, f_out: int, lmax: int , kernel_grid: tuple):
        super().__init__()

        # filter weight parametrized over spatial grid on S2
        self.register_parameter(
          "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]

        # linear projection to convert filter weights to fourier domain
        self.register_buffer(
          "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, (2*lmax+1)**2]


        # defines group convolution using appropriate irreps
        self.lin = o3.Linear( s2_irreps(lmax) , so3_irreps(lmax) ,  f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x , weight=psi)



# # SO(3) Convolution


def so3_healpix_grid(rec_level: int=3):
    """Returns healpix grid over so3 of equally spaced rotations
   
    https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272
    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y
    rec_level | num_points | bin width (deg)
    ----------------------------------------
         0    |         72 |    60
         1    |        576 |    30
         2    |       4608 |    15
         3    |      36864 |    7.5
         4    |     294912 |    3.75
         5    |    2359296 |    1.875
         
    :return: tensor of shape (3, npix)
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    gamma = torch.linspace(0, 2*np.pi, 6*n_side + 1)[:-1]

    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, npix)
    return torch.stack((alpha, beta, gamma)).float()

###convolutation over so3
### maybe faster way to do this
class SO3Conv(torch.nn.Module):
    def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
        super().__init__()

        # filter weight parametrized over spatial grid on SO3
        self.register_parameter(
          "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]

        # wigner D matrices used to project spatial signal to irreps of SO(3)
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, sum_l^L (2*l+1)**2]

        # defines group convolution using appropriate irreps
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        '''Perform SO3 group convolution to produce signal over irreps of SO(3).
        First project filter into fourier domain then perform convolution

        :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
        :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
        '''
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)



# # Loss functions

# In[16]:


def compute_trace(rotA, rotB):
    
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns Tr(rotA, rotB.T)
    '''
    #rotA = rotA.type( torch.long )
    #rotB = rotB.type( torch.long )
    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace

def rotation_error(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns rotation error in radians, tensor of shape (*)
    '''
    #rotA = rotA.type(torch.long)
    #rotB = rotB.type(torch.long)
    trace = compute_trace(rotA, rotB)
    return torch.arccos(torch.clamp( (trace - 1)/2, -1, 1))

def nearest_rotmat(src, target):
    
    '''return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    trace = compute_trace(src.unsqueeze(1), target.unsqueeze(0))
   
    return torch.max(trace, dim=1)[1]




### Induced_I2S network
class Induced_I2S(torch.nn.Module):
    
    ### Instantiate I2S-style network for predicting distributions over SO(3) from
    ### predictions made on single image using an induction layer 
    
    def __init__(self, lmax=9 , kmax = 10 , f_so3 = 300   ):
        
        super().__init__()
        self.lmax = lmax
        self.kmax = kmax
        self.f_so3 = f_so3

        ### no image encoder, can add this later
        ### self.encoder = ImageEncoder()

        ### defining the SO2 action
        SO2_act = gspaces.Rot2dOnR2(N=-1,maximum_frequency=self.kmax)

        #### suppose that output of SWIN is 768 trivial
        rep_in = 768 * [ SO2_act.irrep(0) ]

        self.encoder = ImageEncoder()

        ### 768 copies of trivial rep
        hidden_mulplicities_SO2 = { '0': 768 }

        ### the output mupliciteis of the induced SO3 layer
        ### there are two good options here
        mulplicities_SO3 = {} #{ '0' :1 , '1' :1 , '2' : 1 , '3': 1 , '4':1  , '5':1 , '6':1 }
        for l in range( self.lmax ):
            mulplicities_SO3[ str(l) ] = 1 # (2*l + 1)

        ##### the induction representation layer, 
        ### compute the number of output channels of hidden rep
        channels_in = 768 
        self.img_params = 7
        self.induce = Induction_Layer_III( channels = channels_in, image_shape= self.img_params , k_max =self.kmax, L_max=self.lmax , dict_rep_in = hidden_mulplicities_SO2 , dict_rep_out = mulplicities_SO3 )

        ### output format is: batch, number of output channels, number of input channels
        ### these are all in form of 
        s2_kernel_grid = s2_healpix_grid(max_beta=np.inf, rec_level=1)

        ### THIS IS L_MAX - 1 !!! Need to standardize notations
        ### compute the dimension of s2 input features
        self.f_in = rep_ops.compute_SO3_dimension( mulplicities_SO3 )
        self.s2_conv = S2Conv( f_in = self.f_in , f_out= self.f_so3 , lmax=self.lmax-1 , kernel_grid = s2_kernel_grid )

        #### also L_max - 1 !!! Need to standardize notations
        so3_kernel_grid = so3_healpix_grid(rec_level=3)
        ### output is one dimensional so can use logits
        self.so3_conv = SO3Conv( f_in = self.f_so3 , f_out=1 , lmax=self.lmax-1 , kernel_grid = so3_kernel_grid )
        self.so3_act = e3nn.nn.SO3Activation( self.lmax-1 , self.lmax-1 , act=torch.relu, resolution=20)
        
        output_xyx = so3_healpix_grid(rec_level=2)
        self.register_buffer( "output_wigners", flat_wigner( self.lmax - 1 , *output_xyx).transpose(0,1) )
        self.register_buffer( "output_rotmats", o3.angles_to_matrix(*output_xyx) )


    def forward(self, x):
        
        ###'''Returns so3 irreps
        ###:x: the input image, tensor of shape (B, 1, image_size, image_size)
        ## x must be a geometric tensor

        x = self.encoder(x)    
        x = self.induce( x )

        ### need to include a non-linearity here !!!

        x = self.s2_conv( x )
        x = self.so3_act( x )
        x = self.so3_conv( x )
    
        return x
  
    def compute_loss(self, img, gt_rot):
        ###'''Compute cross entropy loss using ground truth rotation, the correct label
        ###is the nearest rotation in the spatial grid to the ground truth rotation

        ### :img: float tensor of shape (B, 3, 224, 224)
        ### :gt_rotation: valid rotation matrices, tensor of shape (B, 3, 3)
        
        ### run image through network
        x = self.forward(img)
        
        ### make sure output is long type tensor
        # x = x.tensor
        grid_signal = torch.matmul(x, self.output_wigners ).squeeze(1)
        rotmats = self.output_rotmats


        # find nearest grid point to ground truth rotation matrix
        rot_id = nearest_rotmat( gt_rot , rotmats )            
        loss = torch.nn.CrossEntropyLoss()( grid_signal.type( torch.float )  , rot_id  )
        
        with torch.no_grad():
            pred_id = grid_signal.max(dim=1)[1]
            pred_rotmat = rotmats[pred_id]
            acc = rotation_error(gt_rot, pred_rotmat)

        return loss, acc.cpu().numpy()
   
    @torch.no_grad()
    def compute_probabilities(self, img, wigners):
        x = self.forward(img)
        logits = torch.matmul(x, wigners).squeeze(1)
        return torch.nn.Softmax(dim=1)(logits)


############ specifiy the arch
induced_arch = Induced_I2S( lmax=lmax , kmax = kmax , f_so3 = f_out  ).to(device)

#### see if there is loaded model
try:
    long_file = 'l_max_'+str(lmax)+'_'+str(f_out)+'_'+str(learning_rate)+'_'
    path = long_file + 'Encoded_Induced_I2S_SYMSOL.pt'
    induced_arch.load_state_dict( torch.load(path,map_location=device) )
    print("Model Loaded from file!")

except:
    print("No model found on file")


###compute the total number of parameters
torch_total_params = sum( p.numel() for p in induced_arch.parameters() )
pytorch_trainable_params = sum(p.numel() for p in induced_arch.parameters() if p.requires_grad)

print( "Total number of parameters:", torch_total_params)
print( "Total number of trainable parameters:", pytorch_trainable_params )
sys.stdout.flush()

#output_xyx = so3_healpix_grid(rec_level=3).to(device) # 37K points
#output_wigners = flat_wigner( lmax - 1 , *output_xyx).transpose(0, 1).to(device)
#output_rotmats = o3.angles_to_matrix(*output_xyx).to(device)


output_xyx = so3_healpix_grid(rec_level=3) # 37K points
output_wigners = flat_wigner( lmax - 1 , *output_xyx).transpose(0, 1)
output_rotmats = o3.angles_to_matrix(*output_xyx)



### some additional packages
from typing import Optional, List, Callable
import os
import glob
import numpy as np
import torch
import torchvision
from PIL import Image

### SYMSOL CLASS
class SymsolDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 train: bool,
                 set_number: int=1,
                 num_views: int=None,
                ):
        self.mode = 'train' if train else 'test'
        self.path = os.path.join(dataset_path, "symsol", self.mode)
        rotations_data = np.load(os.path.join(self.path, 'rotations.npz'))
        self.class_names = {
            1 : ('tet', 'cube', 'icosa', 'cone', 'cyl' , 'sphereX' , 'tetX' , 'cylO' ),
            2 : ('sphereX',),#, 'cylO', 'sphereX'),
            3 : ('cylO',),#, 'cylO', 'sphereX'),
            4 : ('tetX',),#, 'cylO', 'sphereX'),
        }[set_number]
        self.num_classes = len(self.class_names)

        self.rotations_data = [rotations_data[c][:num_views] for c in self.class_names]
        self.indexers = np.cumsum([len(v) for v in self.rotations_data])

    def __getitem__(self, index):
        cls_ind = np.argmax(index < self.indexers)
        if cls_ind > 0:
            index = index - self.indexers[cls_ind-1]

        rot = self.rotations_data[cls_ind][index]
        # randomly sample one of the valid rotation labels
        rot = rot[np.random.randint(len(rot))]
        rot = torch.from_numpy(rot)

        im_path = os.path.join(self.path, 'images',
                               f'{self.class_names[cls_ind]}_{str(index).zfill(5)}.png')
        img = np.array(Image.open(im_path))
        img = torch.from_numpy(img).to(torch.float32) / 255.
        img = img.permute(2, 0, 1)

        class_index = torch.tensor((cls_ind,), dtype=torch.long)

        return dict(img=img, cls=class_index, rot=rot)

    def __len__(self):
        return self.indexers[-1]

    @property
    def img_shape(self):
        return (3, 224, 224)


### training and testing dataloader
dataset_path = './'
data_set = SymsolDataset( dataset_path , train = True , set_number=1 )
train_dataloader = DataLoader( data_set, batch_size=train_batch_size, shuffle=True)
print("Deep Encoded Induced Train Batch Size:" , train_batch_size)



### training
dataset_path = './'
test_data_set = SymsolDataset( dataset_path ,train=False, set_number=1 )
test_batch_size = 1
test_dataloader = DataLoader( test_data_set, batch_size=test_batch_size, shuffle=True)        


### sample 100 points for testing
sample_test_size = 150


### Training the Induced_I2S Model
### Adam optimizer
optimizer = torch.optim.Adam( induced_arch.parameters(), lr=learning_rate )


cnt = 0
num_epoch = 1
for epoch in range(num_epoch):

    avg_test_acc = 0
    cnt_acc = 0
    for item in train_dataloader:

        x = item['img'].to(device)
        label = item['rot'].to(device)

        optimizer.zero_grad()

        loss , a = induced_arch.compute_loss( x , label )
        loss.backward()
        optimizer.step()

        ### print total model training loss
##        wandb.log( {"train/train_loss": loss} )
        avg_test_acc = avg_test_acc + loss
        cnt_acc = cnt_acc + 1
        
        ### save model and compute sample
        if (cnt%1000 == 0 ):
            avg_test_acc = avg_test_acc/cnt_acc
            wandb.log( {"train/train_loss": avg_test_acc} )
            avg_test_acc = 0
            cnt_acc = 0

            long_file = 'l_max_'+str(lmax)+'_'+str(f_out)+'_'+str(learning_rate)+'_'
            file = long_file + 'Encoded_Induced_I2S_SYMSOL.pt'
            torch.save( induced_arch.state_dict() , file)
            print("Model Saved to file:", cnt)
            sys.stdout.flush()

            ### compute a sample loss
            cnt2 = 0
            induced_arch.eval()
            with torch.no_grad():
                for item in test_dataloader:
                    x = item['img'].to(device)
                    label = item['rot'].to(device)

                    loss , a = induced_arch.compute_loss( x , label )
                    wandb.log( {"sample_test/sample_test_loss": loss} )
                    cnt2 = cnt2 + 1
                    if(cnt2 > sample_test_size):
                        break

            induced_arch.train()
        cnt = cnt + 1

    print()
    print('Epoch Number:' , epoch  )
    sys.stdout.flush()



### 1 test epoch
num_epoch = 1
induced_arch.eval()
with torch.no_grad():
    for item in test_dataloader:

        x = item['img'].to(device)
        label = item['rot'].to(device)

        loss , a = induced_arch.compute_loss( x , label )

        #print()
        #print("Test Loss:" , loss )
        #sys.stdout.flush()
        wandb.log( {"test/test_loss": loss} )



wandb.finish()


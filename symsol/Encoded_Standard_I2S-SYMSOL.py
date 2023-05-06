#!/usr/bin/env python
# coding: utf-8

### Image to Sphere Via Projection For Pose Estimation on SYMSOL


### import relevent packages
import torch
import numpy as np
from e2cnn import gspaces
from e2cnn import nn
from e2cnn import group
from e3nn import o3
import torchvision
import wandb
import random
import datetime

import sys
import e3nn
import represenations_opps as rep_ops
import healpy as hp
import torchvision
import time
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

###import pickle5 as pickle
import pickle


learning_rate = 0.000002
lmax = 5
kmax = 10
sphere_fdim = 1000
f_out = 500
batch_size = 5

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Induced_Rep",
    name=str( datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") ),

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "I2S",
    "dataset": "SYMSOL",
    "lmax": lmax ,
    "kmax": kmax,
    "sphere_fdim" : sphere_fdim,
    "f_out":  f_out ,
    "batch_size": batch_size,
        }
)



# Check for Avalible GPUs
### check cuda read
torch.cuda.is_available()

### use gpus if avalible, otherwise use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### number of gpus avalible
num_gpu = torch.cuda.device_count() 

print("Number of GPUs avalible:", num_gpu )

### get names of gpus if avalible
#name = torch.cuda.get_device_name(0)
#print(name)


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


### convert SO2 reps to e2cnn format
### This should be put in seperate script
def convert_SO2(  input_rep_dict ):
    total_rep = []
    for k in input_rep_dict.keys():
        mulplicites = input_rep_dict[k]        
        total_rep = total_rep + mulplicites * [SO2_act.irrep(int(k))]

    return total_rep


# # Image2Sphere Orthographic Projection Baseline
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

### image 2 sphere orthographic projection
class Image2SphereProjector(torch.nn.Module):
  
    def __init__(self,
               fmap_shape, 
               sphere_fdim: int,
               lmax: int,
               coverage: float = 0.9,
               sigma: float = 0.2,
               max_beta: float = np.radians(90),
               taper_beta: float = np.radians(75),
               rec_level: int = 2,
               n_subset: int = 20,
              ):
        super().__init__()
        self.lmax = lmax
        self.n_subset = n_subset

        # point-wise linear operation to convert to proper dimensionality if needed
        if fmap_shape[0] != sphere_fdim:
          self.conv1x1 = torch.nn.Conv2d(fmap_shape[0], sphere_fdim, 1)
        else:
          self.conv1x1 = torch.nn.Identity()

        # determine sampling locations for orthographic projection
        self.kernel_grid = s2_healpix_grid(max_beta=max_beta, rec_level=rec_level)
        self.xyz = o3.angles_to_xyz(*self.kernel_grid)

        # orthographic projection
        max_radius = torch.linalg.norm(self.xyz[:,[0,2]], dim=1).max()
        sample_x = coverage * self.xyz[:,2] / max_radius # range -1 to 1
        sample_y = coverage * self.xyz[:,0] / max_radius

        gridx, gridy = torch.meshgrid(2*[torch.linspace(-1, 1, fmap_shape[1])], indexing='ij')
        scale = 1 / np.sqrt(2 * np.pi * sigma**2)
        data = scale * torch.exp(-((gridx.unsqueeze(-1) - sample_x).pow(2) \
                                    +(gridy.unsqueeze(-1) - sample_y).pow(2)) / (2*sigma**2) )
        data = data / data.sum((0,1), keepdims=True)

        # apply mask to taper magnitude near border if desired
        betas = self.kernel_grid[1]
        if taper_beta < max_beta:
            mask = ((betas - max_beta)/(taper_beta - max_beta)).clamp(max=1).view(1, 1, -1)
        else:
            mask = torch.ones_like(data)

        data = (mask * data).unsqueeze(0).unsqueeze(0).to(torch.float32)
        self.weight = torch.nn.Parameter(data= data, requires_grad=True)

        self.n_pts = self.weight.shape[-1]
        self.ind = torch.arange(self.n_pts)

        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax+1), *self.kernel_grid, normalization='component')
        )

    def forward(self, x):
        '''
        :x: float tensor of shape (B, C, H, W)
        :return: feature vector of shape (B,P,C) where P is number of points on S2
        '''
        
        #### x.tensor
        x = self.conv1x1(x)

        if self.n_subset is not None:
            self.ind = torch.randperm(self.n_pts)[:self.n_subset]

        x = (x.unsqueeze(-1) * self.weight[..., self.ind]).sum((2,3))
        x = torch.relu(x)
        x = torch.einsum('ni,xyn->xyi', self.Y[self.ind], x) / self.ind.shape[0]**0.5
        return x




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

# In[8]:


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


# # Defining the Standard I2S Network

# In[9]:


### I2S network
class I2S(torch.nn.Module):
    
    ### Instantiate I2S-style network for predicting distributions over SO(3) from
    ### predictions made on single image using an induction layer 
    
    def __init__(self, lmax=10 , kmax = 10 , sphere_fdim = 1000 , f_out = 500 ):
        
        super().__init__()
        self.lmax = lmax
        self.kmax = kmax
        self.sphere_fdim = sphere_fdim
        self.f_out = f_out

        ### no image encoder, can add this later
        self.encoder = ImageEncoder()

        ### defining the SO2 action
        SO2_act = gspaces.Rot2dOnR2(N=-1,maximum_frequency=self.kmax)

        ### suppose that input is trival so2 rep
        rep_in = 768*[ SO2_act.irrep(0) ]

        ##### the induction representation layer, 
        ### compute the number of output channels of hidden rep
        channels_in = 768
        self.img_params = 7
        self.proj = Image2SphereProjector( fmap_shape=(channels_in ,self.img_params ,self.img_params ), sphere_fdim= self.sphere_fdim , lmax=self.lmax-1,
               coverage = 0.9,
               sigma = 0.2,
               max_beta = np.radians(90),
               taper_beta = np.radians(75),
               rec_level = 2,
               n_subset = 20 )
        
        
        ### output format is: batch, number of output channels, number of input channels
        ### these are all in form of 
        s2_kernel_grid = s2_healpix_grid(max_beta=np.inf, rec_level=1)

        ### THIS IS L_MAX - 1 !!! Need to standardize notations
        ### compute the dimension of s2 input features
        ### f_in = rep_ops.compute_SO3_dimension( mulplicities_SO3 )
        self.s2_conv = S2Conv( f_in = self.sphere_fdim , f_out= self.f_out , lmax=self.lmax-1 , kernel_grid = s2_kernel_grid )

        #### also L_max - 1 !!! Need to standardize notations
        so3_kernel_grid = so3_healpix_grid(rec_level=3)
        ### output is one dimensional so can use logits 
        self.so3_conv = SO3Conv( f_in = self.f_out , f_out=1 , lmax=self.lmax-1 , kernel_grid = so3_kernel_grid )
        self.so3_act = e3nn.nn.SO3Activation( self.lmax-1 , self.lmax-1 , act=torch.relu, resolution=20)
        
        output_xyx = so3_healpix_grid(rec_level=2)
        self.register_buffer( "output_wigners", flat_wigner( self.lmax - 1 , *output_xyx).transpose(0,1) )
        self.register_buffer( "output_rotmats", o3.angles_to_matrix(*output_xyx) )


    def forward(self, x):
               
        x = self.encoder(x)
        x = self.proj( x )
        x = self.s2_conv( x )
        x = self.so3_act( x )
        x = self.so3_conv( x )
    
        return x
  
    def compute_loss(self, img, gt_rot):
        
        ### run image through network
        x = self.forward(img)
          
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



### set lmax
standard_i2s = I2S( lmax=lmax , kmax = kmax , sphere_fdim = sphere_fdim , f_out = f_out ).to(device)


#### see if there is loaded model
try:
    long_file = 'l_max_'+str(lmax)+'_'+str(sphere_fdim)+'_'+str(f_out)+'_'+str(learning_rate)+'_'
    path = long_file + 'Encoded_Standard_I2S_SYMSOL.pt'
    standard_i2s.load_state_dict( torch.load(path,map_location=device) )
    print("Model Loaded from file!")

except:
    print("No model found on file")

### save model to file
long_file = 'l_max_'+str(lmax)+'_'+str(sphere_fdim)+'_'+str(f_out)+'_'+str(learning_rate)+'_'
file = long_file + 'Encoded_Standard_I2S_SYMSOL.pt'
torch.save( standard_i2s.state_dict() , file)
print("Model saved to file")

###compute the total number of parameters
torch_total_params = sum( p.numel() for p in standard_i2s.parameters() )
pytorch_trainable_params = sum( p.numel() for p in standard_i2s.parameters() if p.requires_grad)

print( "Total number of parameters:", torch_total_params)
print( "Total number of trainable parameters:", pytorch_trainable_params )
sys.stdout.flush()


##output_xyx = so3_healpix_grid(rec_level=3).to(device) # 37K points
##output_wigners = flat_wigner( lmax - 1 , *output_xyx).transpose(0, 1).to(device)
##output_rotmats = o3.angles_to_matrix(*output_xyx).to(device)

output_xyx = so3_healpix_grid(rec_level=3) # 37K points
output_wigners = flat_wigner( lmax - 1 , *output_xyx).transpose(0, 1)
output_rotmats = o3.angles_to_matrix(*output_xyx)


from typing import Optional, List, Callable
import os
import glob
import numpy as np
import torch
import torchvision
from PIL import Image

####symsol dataset class
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


##### 
dataset_path = './'
data_set = SymsolDataset( dataset_path , train = True , set_number=1   )
train_dataloader = DataLoader( data_set, batch_size=batch_size, shuffle=True)

dataset_path = './'
test_data_set = SymsolDataset( dataset_path , train = False , set_number=1   )
test_dataloader = DataLoader( test_data_set, batch_size=1, shuffle=True)

### sample 100 points for testing
sample_test_size = 150

# print( "Encoded Standard Train Batch Size:" , batch_size ) 
# sys.stdout.flush()


### Adam optimizer
optimizer = torch.optim.Adam( standard_i2s.parameters(), lr=learning_rate )

### watch gradients
### wandb.watch( standard_i2s  , log="all", log_freq=2000 )

cnt = 0
num_epoch = 1
for epoch in range(num_epoch):

    avg_test_acc = 0
    cnt_acc = 0
    for item in train_dataloader:

        x = item['img'].to(device)
        label = item['rot'].to(device)

        optimizer.zero_grad()
       
        loss , a = standard_i2s.compute_loss( x , label )
        loss.backward()
        optimizer.step()

        avg_test_acc = avg_test_acc + loss
        cnt_acc = cnt_acc + 1

    ###  wandb.log({"train/train_loss": loss})
        if (cnt%1000 == 0 ):
            avg_test_acc = avg_test_acc/cnt_acc
            wandb.log({"train/train_loss": avg_test_acc })
            avg_test_acc = 0
            cnt_acc = 0
            long_file ='l_max_'+str(lmax)+'_'+str(sphere_fdim)+'_'+str(f_out)+'_'+str(learning_rate)+'_'
            file = long_file + 'Encoded_Standard_I2S_SYMSOL.pt'
            torch.save( standard_i2s.state_dict() , file)
            print("Model Saved to file:", cnt)
            sys.stdout.flush()
            ### compute a sample loss
            cnt2 = 0
            standard_i2s.eval()
            with torch.no_grad():
                for item in test_dataloader:
                    x = item['img'].to(device)
                    label = item['rot'].to(device)
                    loss , a = standard_i2s.compute_loss( x , label )
                    wandb.log( {"sample_test/sample_test_loss": loss} )
                    cnt2 = cnt2 + 1
                    if(cnt2 > sample_test_size):
                        break

            standard_i2s.train()
        cnt = cnt + 1


    print()
    print('Epoch Number:' , epoch  )
    sys.stdout.flush()



num_epoch = 1
standard_i2s.eval()
with torch.no_grad():
    for item in test_dataloader:

       x = item['img'].to(device)
       label = item['rot'].to(device)

       loss , a = standard_i2s.compute_loss( x , label )
       wandb.log({"test/test_loss": loss})




wandb.finish()

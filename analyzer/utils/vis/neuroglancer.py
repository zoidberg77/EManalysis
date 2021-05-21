import neuroglancer
import numpy as np
import sys
import tifffile
import h5py
# RUN SCRIPT: python3 -i visualng.py
# General settings
ip = 'localhost' # or public IP of the machine for sharable display
port = 13333 # change to an unused port number
neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
#Create a new viewer. This starts a webserver in a background thread, which serves a copy of the Neuroglancer client.
viewer = neuroglancer.Viewer()
file = 'outputs/neuroglancer.h5'
# resolution & dimension of the data
res = [4, 4, 40];
# Adapting viewer config
'''
with viewer.txn() as s:
    s.layout = '3d'
    s.projection_scale = 3000
'''
# Image section
print('load image')
image = h5py.File(file, 'r')
tmp_im = np.array(image['image'][:100, :1000, :1000])
data_im = np.swapaxes(tmp_im, 0, 2)
with viewer.txn() as s:
    s.layers.append(
        name='image',
        layer=neuroglancer.LocalVolume(
            data=data_im,
            #data=np.array(image['main'][:100, :1000, :1000]),
            #voxel_size=res,
            volume_type='image'
        ))
    #s.projection_scale=2000000000
# Segmentation section
print('load segmentation')
image = h5py.File(file, 'r')
tmp_gt = np.array(image['label'][:100, :1000, :1000])
data_gt = np.swapaxes(tmp_gt, 0, 2)
with viewer.txn() as s:
    s.layers.append(
        name = 'segmentation',
        layer = neuroglancer.LocalVolume(
            data = data_gt,
            voxel_size = res,
            volume_type = "segmentation"
        ))
    s.layout = '3d'
print(viewer)
#print(neuroglancer.to_url(viewer.state))
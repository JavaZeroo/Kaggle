import numpy as np
import nibabel as nib
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


image_path = "./test.nii.gz"
image_obj = nib.load(image_path)
print(f'Type of the image {type(image_obj)}')
image_data = image_obj.get_fdata()
type(image_data)
height, width, depth = image_data.shape
print(f"The image object height: {height}, width:{width}, depth:{depth}")
print(f'image value range: [{image_data.min()}, {image_data.max()}]')
print(image_obj.header.keys())
pixdim =  image_obj.header['pixdim']
print(f'z轴分辨率： {pixdim[3]}')
print(f'in plane 分辨率： {pixdim[1]} * {pixdim[2]}')
z_range = pixdim[3] * depth
x_range = pixdim[1] * height
y_range = pixdim[2] * width
print(x_range, y_range, z_range)
maxval = 177
i = np.random.randint(0, maxval)
# Define a channel to look at
print(f"Plotting z Layer {i} of Image")

plt.imshow(image_data[:, :, i], cmap='gray')
plt.axis('off')

def explore_3dimage(layer):
    plt.figure(figsize=(10, 5))
    plt.imshow(image_data[:, :, layer], cmap='gray');
    plt.title('Explore Layers of adrenal', fontsize=20)
    plt.axis('off')
    return layer

interact(explore_3dimage, layer=(0, image_data.shape[-1]));

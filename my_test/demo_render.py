from matplotlib import pyplot as plt
import numpy as np
import torch
import os
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesAtlas, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    Materials,
    HardPhongShader
)
from pytorch3d.renderer.lighting import PointLights
import torchvision.transforms as transforms
from PIL import Image
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def get_uv_map():
    uvs_name = 'utils/NIMBLE_model/assets/faces_uvs.pt'
    faces_uv = torch.load(uvs_name)
    verts_uv = torch.load(uvs_name.replace('faces', 'verts'))
    return faces_uv, verts_uv

def get_224_renderer():
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        # perspective_correct=False, 
    )
    # create a renderer object
    R, T = look_at_view_transform(dist=70, elev=50, azim=-15) 
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    # Create a Materials object with the specified material properties
    # materials = Materials(
    #     ambient_color=((0.2, 0.2, 0.2),),
    #     diffuse_color=((0.8, 0.8, 0.8),),
    #     specular_color=((1, 1, 1),),
    # )
    lighting = PointLights(
                # ambient_color=((0.2, 0.2, 0.2),),
                # diffuse_color=((0.8, 0.8, 0.8),),
                # specular_color=((1, 1, 1),),
                # location=((0.0, 0.0, 0.0),),
                device=device,
            )

    renderer_p3d = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(
            cameras=cameras,
            # materials=materials,
            lights=lighting,
            device=device
        ),
    )
    return renderer_p3d

    
def get_best_renderer():
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=1024, 
        blur_radius=0.0, 
        faces_per_pixel=50, 
        z_clip_value=None
    )
    # create a renderer object

    # Create a Materials object with the specified material properties
    materials = Materials(
        ambient_color=((0.9, 0.9, 0.9),),
        diffuse_color=((0.8, 0.8, 0.8),),
        specular_color=((0.2, 0.2, 0.2),),
        shininess=60,
        device=device,
    )

    renderer_p3d = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings_soft
        ),
        shader=HardPhongShader(
            materials=materials,
            device=device
        ),
    )
    return renderer_p3d

def render(root, filename, renderer):
    # Load the 3D model
    verts, faces, aux = load_obj(os.path.join(root, filename),
                                create_texture_atlas=True, device=device)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images
    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...].to(device)  # (1, H, W, 3)
    # Create a textures object
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)
    # Create a Meshes object
    mesh = Meshes( verts=[verts], faces=[faces.verts_idx], textures=tex)

    # # load diffuse, specular and normal images
    # diffuse_img = Image.open(root + "demo_0006_0000000_diffuse.png").convert("RGB")
    # specular_img = Image.open(root + "demo_0006_0000000_spec.png").convert("RGB")
    # normal_img = Image.open(root + "demo_0006_0000000_normal.png").convert("RGB")

    # # convert images to PyTorch tensors
    # diffuse_tensor = transforms.ToTensor()(diffuse_img).permute(2, 0, 1).unsqueeze(0)
    # specular_tensor = transforms.ToTensor()(specular_img).permute(2, 0, 1).unsqueeze(0)
    # normal_tensor = transforms.ToTensor()(normal_img).permute(2, 0, 1).unsqueeze(0)

    # R, T = look_at_view_transform(dist=100, elev=50, azim=-15) 
    R, T = look_at_view_transform(dist=100) 
    # cameras = PerspectiveCameras(R=R, T=T, in_ndc=False, image_size=((224,224),), device=device)
    cameras = PerspectiveCameras(R=R, T=T, device=device)
    verts_location_mean = tuple(verts.mean(dim=0).tolist())

    lighting = PointLights(
                # ambient_color=((0.2, 0.2, 0.2),),
                # diffuse_color=((0.8, 0.8, 0.8),),
                # specular_color=((1, 1, 1),),
                location=(verts_location_mean,),
                device=device,
            )

    # Render the 3D model with the texture atlas applied
    images = renderer(mesh, cameras=cameras, lighting=lighting, znear=-2, zfar=1000.0)
    return images

def pyplot_save_img(img: torch.Tensor, save_path):
    if img.dim() == 4:
        img = img.squeeze(0)
    plt.figure()
    plt.imshow(img.cpu().detach().numpy())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    root = '/home/jiayin/HandRecon/utils/NIMBLE_model/output/'
    renderer = get_best_renderer()
    file_list = os.listdir(root)
    obj_files = [file for file in file_list if file.endswith('skin.obj')]
    for file in obj_files:
        image = render(root, file, renderer)
        pyplot_save_img(image[..., :3], root+file[:-4]+'_rendered_pyplot.png')
        # normalize the tensor values between 0 and 1
        image_normalized = (image - image.min()) / (image.max() - image.min())
        # convert the tensor to a PIL image
        image = transforms.ToPILImage()(image_normalized.squeeze()[..., :3].permute(2, 0, 1))
        save_path = root+file[:-4]+'_rendered.png'
        image.save(save_path)
        print('save image to: ', save_path)
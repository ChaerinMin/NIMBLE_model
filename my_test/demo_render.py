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
    Materials
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

def get_renderer():
    sigma = 1e-4
    raster_settings_soft = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        # perspective_correct=False, 
    )
    # create a renderer object
    R, T = look_at_view_transform(dist=70, elev=50, azim=-15) 
    cameras = FoVPerspectiveCameras(R=R, T=T)

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
                # device=device,
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
        ),
    )
    return renderer_p3d

def render(root, filename, renderer):
    # Load the 3D model
    verts, faces, aux = load_obj(os.path.join(root, filename),
                                create_texture_atlas=True)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    torch.save(verts_uvs, root+'verts_uvs.pt')
    torch.save(faces_uvs, root+'faces_uvs.pt')
    tex_maps = aux.texture_images
    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)
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

    # Render the 3D model with the texture atlas applied
    images = renderer(mesh)
    # normalize the tensor values between 0 and 1
    image_normalized = (images - images.min()) / (images.max() - images.min())
    # convert the tensor to a PIL image
    image = transforms.ToPILImage()(image_normalized.squeeze()[..., :3].permute(2, 0, 1))
    return image
    # save the image
    # image.save(root+'image.png')

if __name__ == '__main__':
    root = '/home/jiayin/HandRecon/utils/NIMBLE_model/output/'
    renderer = get_renderer()
    file_list = os.listdir(root)
    obj_files = [file for file in file_list if file.endswith('skin.obj')]
    for file in obj_files:
        image = render(root, file, renderer)
        save_path = root+file[:-4]+'_rendered.png'
        image.save(save_path)
        print('save image to: ', save_path)
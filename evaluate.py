import os
import cfg
from sam import sam_model_registry

import torch
import torchio as tio
import numpy as np
from torchvision import transforms

# Hyperparameter for SegmentAnyMuscle
args = cfg.parse_args()
args.if_mask_decoder_adapter=True
args.if_encoder_adapter = True
args.decoder_adapt_depth = 2

moe = 10
k = 10

device = 'cuda' #'cpu' if you want to run on cpu

# Define model
model = sam_model_registry["vit_t"](args, checkpoint=None, num_classes=3, moe=moe, k=k)
model.load_state_dict(torch.load('checkpoints/finetuned_sam.pth'), strict=True)
model = model.to(device).eval()

# Load image folder and define output folder
volume_dir = 'TEST_FOLDER'
output_dir = 'TEST_FOLDER_OUT'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

data_list = os.listdir(volume_dir)
for i in range(len(data_list)):
    img_name = data_list[i]
    image_vol = tio.ScalarImage(os.path.join(volume_dir,img_name))

    idx_list = list(range(image_vol.shape[3]))
    mask_vol_numpy = np.zeros(image_vol.shape)

    for idx in idx_list:
        img = image_vol.data[:,:,:,idx].repeat(3,1,1)
        img = transforms.Resize((1024,1024))(img)
        img = (img-img.min())/(img.max()-img.min()+1e-8) 
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)   
        img = img.float().to(device)

        # model forward
        img_emb, _, _ = model.image_encoder(img.unsqueeze(0), None)
        sparse_emb, dense_emb = model.prompt_encoder(
                                    points=None,
                                    boxes=None,
                                    masks=None,
                                )
        pred_auto, _, _, _ = model.mask_decoder(
                                image_embeddings=img_emb,
                                image_pe=model.prompt_encoder.get_dense_pe(), 
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb, 
                                multimask_output=True,
                        )
        pred_auto = pred_auto[:,1,:]

        pred_output = transforms.Resize((image_vol.shape[1], image_vol.shape[2]))(pred_auto)
        pred_output = pred_output >= 0
        mask_vol_numpy[:,:,:,idx] = pred_output.cpu().detach().numpy()
    
    np.save(os.path.join(output_dir, img_name.replace('.nii.gz', '.npy')), mask_vol_numpy)
    # Optionally if you want to save the output as seg.nrrd format
    mask_vol = tio.LabelMap(tensor=torch.tensor(mask_vol_numpy,dtype=torch.int), affine=image_vol.affine)
    mask_vol.save(os.path.join(output_dir, img_name.replace('.nii.gz','.seg.nrrd')))

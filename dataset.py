import os, torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import random
import torchio as tio
import slicerio
import nrrd
import monai
import pickle
import nibabel as nib
from scipy.ndimage import zoom
from monai.transforms import OneOf
import einops
#from funcs import *
from torchvision.transforms import InterpolationMode
#from .utils.transforms import ResizeLongestSide


class MRI_dataset(Dataset):
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',sample_num=50,channel_num=1,crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_15',prompt_num=15,delete_empty_masks=False,if_attention_map=None):
        super(MRI_dataset, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        self.if_attention_map = if_attention_map
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
            
        namefiles = open(img_list,'r')
        self.data_list = namefiles.read().split('\n')[:-1]

        if delete_empty_masks=='delete' or delete_empty_masks=='subsample':
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                mask_path = data.split(' ')[1]
                if os.path.exists(os.path.join(self.mask_folder,mask_path)):
                    msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                else:
                    msk = Image.open(os.path.join(self.mask_folder.replace('2D-slices','2D-slices-generated'),mask_path)).convert('L')
                if 'all' in self.targets: # combine all targets as single target
                    mask_cls = np.array(np.array(msk,dtype=int)>0,dtype=int)
                else:
                    mask_cls = np.array(msk==self.cls,dtype=int)
                if part_list[0]=='all' and np.sum(mask_cls)>0:
                    keep_idx.append(idx) 
                elif np.sum(mask_cls)>0:
                    if_keep = False
                    for part in part_list:
                        if mask_path.find(part)>=0:
                            if_keep = True
                    if if_keep:
                        keep_idx.append(idx) 
            print('num with non-empty masks',len(keep_idx),'num with all masks',len(self.data_list))  
            if delete_empty_masks=='subsample':
                empty_idx = list(set(range(len(self.data_list)))-set(keep_idx))
                keep_empty_idx = random.sample(empty_idx, int(len(empty_idx)*0.1))
                keep_idx = empty_idx + keep_idx
            self.data_list = [self.data_list[i] for i in keep_idx] # keep the slices that contains target mask
  
        if phase == 'train':
            self.aug_img = [transforms.RandomEqualize(p=0.1),
                             transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                             transforms.RandomAdjustSharpness(0.5, p=0.5),
                             ]
            self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.2)),
                     transforms.RandomRotation(45)])
            transform_img = [transforms.ToTensor()]
        else:
            transform_img = [
                         transforms.ToTensor(),
                             ]
        self.transform_img = transforms.Compose(transform_img)
            
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(' ')[0]
        mask_path = data.split(' ')[1]
        slice_num = data.split(' ')[3] # total slice num for this object
        #print(img_path,mask_path)
        try:
            if os.path.exists(os.path.join(self.img_folder,img_path)):
                img = Image.open(os.path.join(self.img_folder,img_path)).convert('RGB')
            else:
                img = Image.open(os.path.join(self.img_folder.replace('2D-slices','2D-slices-generated'),img_path)).convert('RGB')
        except:
            # try to load image as numpy file
            img_arr = np.load(os.path.join(self.img_folder,img_path)) 
            img_arr = np.array((img_arr-img_arr.min())/(img_arr.max()-img_arr.min()+1e-8)*255,dtype=np.uint8)
            img_3c = np.tile(img_arr[:, :,None], [1, 1, 3])
            img = Image.fromarray(img_3c, 'RGB')
        if os.path.exists(os.path.join(self.mask_folder,mask_path)):
            msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
        else:
            msk = Image.open(os.path.join(self.mask_folder.replace('2D-slices','2D-slices-generated'),mask_path)).convert('L')
                    
        if self.if_attention_map:
            slice_id = int(img_path.split('-')[-1].split('.')[0])
            slice_fraction = int(slice_id/int(slice_num)*4)
            img_id = '/'.join(img_path.split('-')[:-1]) +'_'+str(slice_fraction) + '.npy'
            attention_map = torch.tensor(np.load(os.path.join(self.if_attention_map,img_id)))
        else:
            attention_map = torch.zeros((64,64))
        
        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        state = torch.get_rng_state()
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = transforms.functional.pad(img, padding, 0, 'constant')
            torch.set_rng_state(state)
            t,l,h,w=transforms.RandomCrop.get_params(img,(self.crop_size,self.crop_size))
            img = transforms.functional.crop(img, t, l, h,w) 
            msk = transforms.functional.pad(msk, padding, 0, 'constant')
            msk = transforms.functional.crop(msk, t, l, h,w)
        if self.phase =='train':
            # add random optimazition
            aug_img_fuc = transforms.RandomChoice(self.aug_img)
            img = aug_img_fuc(img)

        img = self.transform_img(img)
        if self.phase == 'train':
            # It will randomly choose one
            random_transform = OneOf([monai.transforms.RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),\
                                      monai.transforms.RandKSpaceSpikeNoise(prob=0.5, intensity_range=None, channel_wise=True),\
                                      monai.transforms.RandBiasField(degree=3),\
                                      monai.transforms.RandGibbsNoise(prob=0.5, alpha=(0.0, 1.0))
                                     ],weights=[0.3,0.3,0.2,0.2])
            img = random_transform(img).as_tensor()
        else:
            if img.mean()<0.05:
                img = min_max_normalize(img)
                img = monai.transforms.AdjustContrast(gamma=0.8)(img)

        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
            
        mask_cls = np.array(msk==self.cls,dtype=int)

        if self.phase=='train' and (not self.if_attention_map==None):
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
            both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int)

        img = (img-img.min())/(img.max()-img.min()+1e-8)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        # generate mask and prompt
        if self.if_prompt:
            if self.prompt_type =='point':
                prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'point_coords': pc,
                    'point_labels':pl,
                    'img_name':img_path,
                    'atten_map':attention_map,
            }
            elif self.prompt_type =='box':
                prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                box = torch.as_tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'boxes':box,
                    'img_name':img_path,
                    'atten_map':attention_map,
            }
        else:
            msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            return {'image':img,
                'mask':msk,
                'img_name':img_path,
                'atten_map':attention_map,
        }


class MRI_dataset_multicls(Dataset):
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',\
                 channel_num=1,image_size=1024,targets=['combine_all'], \
                 if_prompt=False,prompt_type='point',region_type='largest_20',prompt_num=20, \
                 delete_empty_masks=False,label_mapping=None, \
                 random_aug=False, image_type=None, seq_type=None):
        '''
        target: 'combine_all':combine all targets into binary;
                'multi_all':give a multi-cls mask output
                'hip/femur':select single target name;
                ’random': choose random cls for each item;
        '''
        super(MRI_dataset_multicls, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.args = args
        self.random_aug = random_aug

        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        self.label_dic = {}
        
        self.label_name_list = []
        if label_mapping:
            with open(label_mapping, 'rb') as handle:
                self.segment_names_to_labels = pickle.load(handle)
            for seg in self.segment_names_to_labels:
                if not seg[1] in self.label_dic.keys():
                    self.label_dic[seg[1]]=seg[0]
                self.label_name_list.append(seg[0])
            print(self.label_dic)
        else:
            self.label_dic = {}
            for value in range(1,256):
                self.label_dic[value] = 'all'
        
        if self.targets[0] in self.label_name_list:
            for seg in self.segment_names_to_labels:
                if seg[0] in targets: 
                    self.cls = int(seg[1])
            
        namefiles = open(img_list,'r')
        self.data_list = namefiles.read().split('\n')[:-1]
        
        if ',' in self.data_list[0]:
            self.sp_symbol = ','
        else:
            self.sp_symbol = ' '
    
        
        if delete_empty_masks: 
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                img_path = data.split(self.sp_symbol)[0]
                if img_path.startswith('/'):
                    img_path = img_path[1:]   
                mask_path = data.split(self.sp_symbol)[1]
                if mask_path.startswith('/'):
                    mask_path = mask_path[1:]


                msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                if 'combine_all' in self.targets: # combine all targets as single target
                    mask_cls = np.array(np.array(msk,dtype=int)>0,dtype=int)
                elif self.targets[0] in self.label_name_list:
                    mask_cls = np.array(np.array(msk,dtype=int)==self.cls,dtype=int)
                else:
                    mask_cls = np.array(msk,dtype=int)

                if np.sum(mask_cls)>0:
                    keep_idx.append(idx) 


            self.data_list = [self.data_list[i] for i in keep_idx] # keep the slices that contains target mask
            print('num with non-empty masks',len(keep_idx), 'num with all masks',len(self.data_list))        
        else:
            print('num with all masks', len(self.data_list))

        if image_type is not None:
            tmp = []
            for d in self.data_list:
                part = d.split(' ')[1].split('_')[0]
                # TODO: fix this; registered image, incorrect format
                if part == 'Patient':
                    part = 'abdm'

                if 'thoracic_pt3_ex1_vol14' in d or 'thoracic_pt2_ex1_vol6' in d:
                    part = 'lumbar'
                if 'pelvis_pt18_ex0_vol5' in d:
                    part = 'abdm'
                if 'thigh_pt14_ex0_vol6' in d:
                    part = 'hip'

                if part in image_type:
                    tmp.append(d)
            self.data_list = tmp
            print('filter data by', image_type, 'num left', len(self.data_list))


        if phase == 'train':
            self.aug_img = [transforms.RandomEqualize(p=0.1),
                            transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                            transforms.RandomAdjustSharpness(0.5, p=0.5),
                           ]
            self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(image_size, scale=(0.3, 2)),
                     transforms.RandomRotation(45)])

        transform_img = [transforms.ToTensor()]    
        self.transform_img = transforms.Compose(transform_img)
       
        def convert_name(img_name):
            if 'mnt' not in img_name:
                part = img_name.split('_')[0]
                if part == 'lowerleg':
                    part = 'lower_leg'
                if part == 'abdomen':
                    part = 'abdm'

                return part + '_' + img_name.split('_')[1] + img_name.split('_')[2]  + '_' + img_name.split('_')[3].split('-')[0]
            else:
                part = img_name.split('/')[-4]
                if part == 'lowerleg':
                    part = 'lower_leg'
                if part == 'abdomen':
                    part = 'abdm'

                return part + '_' + img_name.split('/')[-3].replace('Patient_', 'pt') + 'ex' + img_name.split('/')[-2].replace('Exam_', '')  + '_' + img_name.split('/')[-1].split('-')[0]

        if seq_type is not None:
            seq_type = [s.replace('seq_', '') for s in seq_type]

            patient2seq = {}
            df = pd.read_csv('test_scripts/Phase1_MR_addseqgroup.csv')
            for i in range(len(df)):
                df_curr = df.iloc[i]
                pid = df_curr['Patient'].split('_')[-1]
                eid = df_curr['Exam'].split('_')[-1]
                vid = df_curr['Vol'].split('.')[0]
                part = df_curr['fake part']
                if part == 'abdomen':
                    part = 'abdm'
                part = part.replace('_spine', '')
                name = '%s_pt%sex%s_%s' % (part, pid, eid, vid)
                patient2seq[name] = df_curr['Series Category']

            df = pd.read_csv('test_scripts/Phase2and3_MR_onlykept_afterconvert3D_addseqgroup.csv')
            for i in range(len(df)):
                df_curr = df.iloc[i]
                pid = df_curr['Patient ID']
                eid = df_curr['Exam ID']
                vid = df_curr['save_path'].split('/')[-1].split('.')[0]
                part = df_curr['real part']
                if part == 'abdomen':
                    part = 'abdm'
                part = part.replace('_spine', '')
                name = '%s_pt%sex%s_%s' % (part, pid, eid, vid)
                patient2seq[name] = df_curr['Series Category']

            tmp = []
            for d in self.data_list:
                patient_name = d.split(' ')[0]
                curr_seq = patient2seq[convert_name(patient_name)]
                if curr_seq in seq_type:
                    tmp.append(d)
            self.data_list = tmp
            print('filter data by', seq_type, 'num left', len(self.data_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(self.sp_symbol)[0]
        mask_path = data.split(self.sp_symbol)[1]
        if_paired = data.split(self.sp_symbol)[-1]

        if mask_path.startswith('/'):
            mask_path = mask_path[1:]
        if img_path.startswith('/'):
            img_path = img_path[1:]

        #print(img_path,mask_path)
        if os.path.exists(os.path.join(self.img_folder,img_path)):
            img = Image.open(os.path.join(self.img_folder,img_path)).convert('RGB')
            msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
        else:
            # try to load image from another folder
            img_folder_backup = self.img_folder.replace('public_data','private_data')
            msk_folder_backup = self.mask_folder.replace('public_data','private_data')
            img = Image.open(os.path.join(img_folder_backup,img_path)).convert('RGB')
            msk = Image.open(os.path.join(msk_folder_backup,mask_path)).convert('L')
        
        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        if self.phase =='train':
            # add random optimazition for img
            aug_img_fuc = transforms.RandomChoice(self.aug_img)
            img = aug_img_fuc(img)
            img = self.transform_img(img)
            img = monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1)(img)
            img = monai.transforms.RandGibbsNoise(prob=0.1, alpha=(0.0, 1.0))(img).as_tensor()
        elif self.phase =='val' or self.phase == 'test':
            img = self.transform_img(img)
        
        msk = np.array(msk,dtype=int) 
        unique_classes = np.unique(msk).tolist()
        unique_classes.remove(0)
        if len(unique_classes)>0:
            selected_dic = dict((k,self.label_dic[k]) for k in unique_classes if k in self.label_dic)
        else:
            selected_dic = {}
        
        if 'combine_all' in self.targets: # combine all targets as single target
            mask_cls = np.array(np.array(msk>0),dtype=int)
        elif self.targets[0] in self.label_name_list:
            mask_cls = np.array(msk==self.cls,dtype=int)
        else:
            mask_cls = np.array(msk,dtype=int)

        if self.phase=='train':
            if len(img.shape)==3:
                mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
                both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            else:
                mask_cls = np.repeat(mask_cls,3,axis=1)
                both_targets = torch.cat((img, torch.tensor(mask_cls)),0)
            transformed_targets = self.transform_spatial(both_targets)

            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[-1][0].detach(),dtype=int)
            
        if self.random_aug:
            random_scale = torch.rand(1).item()

            #random_scale = (random_scale - 0.5) / 2
            #img = img + random_scale * mask_cls

            #random_scale = (random_scale - 0.5) / 2
            #img = img + random_scale

            random_scale = random_scale / 10 * 3
            img = img * random_scale
                
            img = torch.clip(img, min=0, max=1)
            img = img.float()

        img = (img-img.min())/(img.max()-img.min()+1e-8)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        if self.if_prompt:
            if self.prompt_type =='point':
                gen_prompt_function = get_first_prompt
            else:
                gen_prompt_function = get_top_boxes
            if len(mask_cls.shape)==3: # multi-slice setting
                prompt= []
                mask_now = []
                for slice_idx in range(mask_cls.shape[0]):
                    prompt_slice,mask_slice = gen_prompt_function(mask_cls[slice_idx],region_type=self.region_type,prompt_num=self.prompt_num)
                    prompt.append(prompt_slice[np.newaxis,:])
                    mask_now.append(mask_slice[np.newaxis,:])
                prompt = np.concatenate(prompt,0)
                mask_now = np.concatenate(mask_now,0)
            else:
                prompt,mask_now = gen_prompt_function(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                        
            if len(mask_now.shape)==2:
                mask_now = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                mask_cls = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            elif len(mask_cls.shape)==4:
                msk = torch.squeeze(torch.tensor(mask_cls,dtype=torch.long))
            else:
                mask_now = torch.tensor(mask_now,dtype=torch.long)
                mask_cls = torch.tensor(mask_cls,dtype=torch.long)
            ref_msk,_ = torch.max(mask_now>0,dim=0)

            if self.prompt_type =='point':
                pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                
                return {'image':img,
                    'mask':mask_now,
                    'selected_label_name':selected_label,
                    'cls_one_hot':cls_one_hot,
                    'point_coords': pc,
                    'point_labels':pl,
                    'img_name':img_path,
                    'mask_ori':msk,
                    'mask_cls':mask_cls,
                    'ref_mask':ref_msk,

            }
            elif self.prompt_type =='box':
                box = torch.as_tensor(prompt, dtype=torch.float)
                #print(box.shape)
                return {'image':img,
                    'mask':mask_now,
                    'selected_label_name':selected_label,
                    'boxes':box,
                    'img_name':img_path,
                    'mask_ori':msk,
                    'mask_cls':mask_cls,
                    'cls_one_hot':cls_one_hot,
                    'all_label_dic':label_annotated,
                    'ref_mask':ref_msk,
                    'atten_map':attention_map,
                    

            }
        else:
            if len(mask_cls.shape)==2:
                msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            elif len(mask_cls.shape)==4:
                msk = torch.squeeze(torch.tensor(mask_cls,dtype=torch.long))
            else:
                msk = torch.tensor(mask_cls,dtype=torch.long)
            return {'image':img,
                    'mask':msk,
                    'img_name':img_path,
                    'mask_cls':msk,
                    'if_paired':if_paired,
            }

        
    
class MRI_dataset_ref(Dataset):
    '''
    this is the multi-slice version dataset
    '''
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',sample_num=50,channel_num=1,crop=False,crop_size=1024,targets=['all'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_10',prompt_num=10):
        super(MRI_dataset_ref, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.reference_slices = {}
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
        #self.img_list = [i for i in os.listdir(self.imgs_dir) if not i.startswith('.')]
        if phase == 'train':
            namefiles = open(img_list,'r')
            self.data_list = namefiles.read().split('\n')[:-1]
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                img_path = data.split(' ')[0]
                msk_path = data.split(' ')[1]
                slice_num = data.split(' ')[2]
                volume_name = ''.join(img_path.split('-')[:-1]) # get volume name
                
                if not (volume_name in self.reference_slices.keys()):
                    self.reference_slices[volume_name]=[(img_path,msk_path,slice_num)]
                else:
                    self.reference_slices[volume_name]+=[(img_path,msk_path,slice_num)]
             
                    
            #self.data_list = [self.data_list[i] for i in keep_idx] # keep the slices that contains target mask
  
        elif phase == 'val':
            namefiles = open(img_list,'r')
            self.data_list = namefiles.read().split('\n')[:-1]
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                img_path = data.split(' ')[0]
                msk_path = data.split(' ')[1]
                slice_num = data.split(' ')[2]
                volume_name = ''.join(img_path.split('-')[:-1]) # get volume name
                if not (volume_name in self.reference_slices.keys()):
                    self.reference_slices[volume_name]=[(img_path,msk_path,slice_num)]
                else:
                    self.reference_slices[volume_name]+=[(img_path,msk_path,slice_num)]

        if phase == 'train':
            self.aug_img = [transforms.RandomEqualize(p=0.1),
                             transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                             transforms.RandomAdjustSharpness(0.5, p=0.5),
                             ]
            self.transform_spatial = transforms.Compose([transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.2)),
                     transforms.RandomRotation(45)])
            transform_img = [transforms.ToTensor()]
        else:
            transform_img = [
                         transforms.ToTensor(),
                             ]
        self.transform_img = transforms.Compose(transform_img)
            
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(' ')[0]
        mask_path = data.split(' ')[1]
        slice_num = int(data.split(' ')[2])
        #print(img_path,mask_path)
        volume_name = ''.join(img_path.split('-')[:-1])
        
        reference_slices = self.reference_slices[volume_name]
        ori_reference_slices = reference_slices.copy()
        if 0<slice_num<len(reference_slices)-1:
            reference_slices = reference_slices[slice_num-1:slice_num+2]
        elif slice_num==0:
            reference_slices = reference_slices[0:2]
            reference_slices.insert(0,ori_reference_slices[0])
        else:
            reference_slices = reference_slices[-2:]
            reference_slices.append(ori_reference_slices[-1])
            
        ref_slices = []
        for ref_slice in reference_slices:
            #print(ref_slice)
            ref_img_path = ref_slice[0]
            ref_img = Image.open(os.path.join(self.img_folder,ref_img_path)).convert('L')
            ref_img = transforms.Resize((self.args.image_size,self.args.image_size))(ref_img)
            ref_img = self.transform_img(ref_img)
            #ref_img = torch.unsqueeze(ref_img,0)
            ref_slices.append(ref_img)
        img = torch.cat(ref_slices,dim=0)
                 
            
        if self.phase == 'train':
            img = monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1)(img)
            img = monai.transforms.RandGibbsNoise(prob=0.1, alpha=(0.0, 1.0))(img).as_tensor()
        msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
        mask_cls = np.array(msk==self.cls,dtype=int)
        
        state = torch.get_rng_state()
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = transforms.functional.pad(img, padding, 0, 'constant')
            torch.set_rng_state(state)
            t,l,h,w=transforms.RandomCrop.get_params(img,(self.crop_size,self.crop_size))
            img = transforms.functional.crop(img, t, l, h,w) 
            msk = transforms.functional.pad(msk, padding, 0, 'constant')
            msk = transforms.functional.crop(msk, t, l, h,w)
    
        # generate mask and prompt
        if self.phase=='train':
            mask_cls = np.repeat(mask_cls[np.newaxis,:, :], 3, axis=0)
            both_targets = torch.cat((img.unsqueeze(0), torch.tensor(mask_cls).unsqueeze(0)),0)
            transformed_targets = self.transform_spatial(both_targets)
            img = transformed_targets[0]
            mask_cls = np.array(transformed_targets[1][0].detach(),dtype=int)
        
        img = (img-img.min())/(img.max()-img.min())
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        try:
            # generate mask and prompt
            if self.if_prompt:
                if self.prompt_type =='point':
                    prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                    pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                    pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                    msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                    return {'image':img,
                        'mask':msk,
                        'point_coords': pc,
                        'point_labels':pl,
                        'img_name':img_path,
                }
                elif self.prompt_type =='box':
                    prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type)
                    box = torch.as_tensor(prompt, dtype=torch.float)
                    msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                    return {'image':img,
                        'mask':msk,
                        'boxes':box,
                        'img_name':img_path,
                }
            else:
                msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'img_name':img_path,
            }
        except:
            print('Errors happen on:',img_path)
        
        
        
class Public_dataset(Dataset):
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',sample_num=50,channel_num=1,crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_3'):
        '''
        cls: the target cls for segmentation
        prompt_type: point or box
        '''
        super(Public_dataset, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
            
        if phase == 'train':
            namefiles = open(img_list,'r')
            self.data_list = namefiles.read().split('\n')[:-1]
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                mask_path = data.split(',')[1]
                if mask_path.startswith('/'):
                    mask_path = mask_path[1:]
                msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                mask_cls = np.array(msk,dtype=int)
                if part_list[0]=='all' and np.sum(mask_cls)>0:
                    keep_idx.append(idx) 
                elif np.sum(mask_cls)>0:
                    if_keep = False
                    for part in part_list:
                        if mask_path.find(part)>=0:
                            if_keep = True
                    if if_keep:
                        keep_idx.append(idx) 
            print('num with non-empty masks',len(keep_idx),'num with all masks',len(self.data_list))        
            self.data_list = [self.data_list[i] for i in keep_idx] # keep the slices that contains target mask
  
        elif phase == 'val':
            namefiles = open(img_list,'r')
            self.data_list = namefiles.read().split('\n')[:-1]
            keep_idx = []
            for idx,data in enumerate(self.data_list):
                mask_path = data.split(',')[1]
                if mask_path.startswith('/'):
                    mask_path = mask_path[1:]
                msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
                #mask_cls = np.array(np.array(msk,dtype=int)==self.cls,dtype=int)
                mask_cls = np.array(msk,dtype=int)
                if part_list[0]=='all' and np.sum(mask_cls)>0:
                    keep_idx.append(idx) 
                elif np.sum(mask_cls)>0:
                    if_keep = False
                    for part in part_list:
                        if mask_path.find(part)>=0:
                            if_keep = True
                    if if_keep:
                        keep_idx.append(idx) 
            print('num with non-empty masks',len(keep_idx),'num with all masks',len(self.data_list))
            self.data_list = [self.data_list[i] for i in keep_idx]

        if phase == 'train':
            transform_img = [transforms.RandomEqualize(p=0.1),
                             transforms.ColorJitter(brightness=0.3, contrast=0.3,saturation=0.3,hue=0.3),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             ]
        else:
            transform_img = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                             ]
        self.transform_img = transforms.Compose(transform_img)
            
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(',')[0]
        mask_path = data.split(',')[1]
        #print(img_path,mask_path)
        img = Image.open(os.path.join(self.img_folder,img_path)).convert('RGB')
        if mask_path.startswith('/'):
            mask_path = mask_path[1:]
        msk = Image.open(os.path.join(self.mask_folder,mask_path)).convert('L')
        
        img = transforms.Resize((self.args.image_size,self.args.image_size))(img)
        msk = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk)
        
        state = torch.get_rng_state()
        if self.crop:
            im_w, im_h = img.size
            diff_w = max(0,self.crop_size-im_w)
            diff_h = max(0,self.crop_size-im_h)
            padding = (diff_w//2, diff_h//2, diff_w-diff_w//2, diff_h-diff_h//2)
            img = transforms.functional.pad(img, padding, 0, 'constant')
            torch.set_rng_state(state)
            t,l,h,w=transforms.RandomCrop.get_params(img,(self.crop_size,self.crop_size))
            img = transforms.functional.crop(img, t, l, h,w) 
            msk = transforms.functional.pad(msk, padding, 0, 'constant')
            msk = transforms.functional.crop(msk, t, l, h,w)
        img = self.transform_img(img)
        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
            
        mask_cls = np.array(msk==self.cls,dtype=int)
        
        
        # generate mask and prompt
        if self.if_prompt:
            if self.prompt_type =='point':
                prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type)
                pc = torch.as_tensor(prompt[:,:2], dtype=torch.float)
                pl = torch.as_tensor(prompt[:, -1], dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'point_coords': pc,
                    'point_labels':pl,
                    'img_name':img_path,
            }
            elif self.prompt_type =='box':
                prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type)
                box = torch.as_tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'boxes':box,
                    'img_name':img_path,
            }
        else:
            msk = torch.unsqueeze(torch.tensor(mask_cls,dtype=torch.long),0)
            return {'image':img,
                'mask':msk,
                'img_name':img_path,
        }

        
        
class MRI_dataset_3D(Dataset):
    '''
    It is a dataset for reading 3D volumes
    if choose if_prompt = False: it is for automatically segmenation
        image_input: shape is 1*d*3*w*h: d is depth of volume, and 3 means 3 channel for each slice.
    '''
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',channel_num=1,crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_10',prompt_num=10,if_attention_map=None):
        super(MRI_dataset_3D, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        self.depth = self.args.depth
        self.if_attention_map = if_attention_map
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
            
        namefiles = open(img_list,'r')
        self.data_list = namefiles.read().split('\n')[:-1]

        aug_transforms = {
                        tio.transforms.RandomElasticDeformation():0.2,
                        tio.transforms.RandomAffine():0.3,
                        tio.transforms.RandomMotion():0.1,
                        tio.transforms.RandomGhosting():0.1,
                        tio.transforms.RandomBiasField():0.1,
                        tio.transforms.RandomBlur():0.1,
                        tio.transforms.RandomNoise():0.1,
            
                         }
        train_transforms = tio.Compose([tio.OneOf(aug_transforms, p = 0.995),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                            ])
        eval_transforms = tio.RescaleIntensity(out_min_max=(0, 1))
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
            
    def __len__(self):
        return len(self.data_list)
    
    def _augment_data(self,data):
        if self.phase == 'train':
            data = self.train_transforms(data)
        else:
            data = self.eval_transforms(data)
        return data
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(' ')[0]
        mask_path = data.split(' ')[1]
        #print(img_path,mask_path)
        state = torch.get_rng_state()
        
        img = nib.load(os.path.join(self.img_folder,img_path))
        
        
        if self.if_attention_map:
            img_id = (img_path).replace('.nii.gz','.npy')
            attention_map = torch.tensor(np.load(os.path.join(self.if_attention_map,img_id)))
        else:
            attention_map = torch.zeros((64,64))
            
            
        img_array = np.expand_dims(np.array(img.dataobj),axis=0) #1*w*h*d
        
        img_array = (img_array-img_array.min())/(img_array.max()-img_array.min()+1e-8)
        mask = nib.load(os.path.join(self.mask_folder,mask_path)) # 1*w*h*d
        msk= np.expand_dims(np.array(mask.dataobj),axis=0)
        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
            
        #extract the target objects
        mask_array = np.array(msk==self.cls,dtype=int)
        
        
        if 0<self.depth<msk.shape[3]:
            # the depth we want to sample is smaller than original object depth
            sample_step = int(msk.shape[3]/self.depth)
            img_array = img_array[:,:,:,::sample_step]
            mask_array =  mask_array[:,:,:,::sample_step]
        elif self.depth<0:
            random_slice = random.randint(0,msk.shape[3]-1)
            img_array = img_array[:,:,:,random_slice:random_slice+1]
            mask_array = mask_array[:,:,:,random_slice:random_slice+1]
        #print(os.path.join(self.img_folder,img_path),img_array.shape,mask_array.shape)
        obj = tio.Subject(
                    image = tio.ScalarImage(tensor = img_array),
                    mask = tio.LabelMap(tensor = mask_array)
                )
        
        aug_obj = self._augment_data(obj)
        
        img_volume = aug_obj.image[tio.DATA].float()  # 1*w*h*d
        #print(img_volume.max())
        msk_volume = aug_obj.mask[tio.DATA].float() 
        
        img_volume = img_volume.permute(0,3,1,2) # 1*d*w*h
        _,d,w,h = img_volume.shape
        msk_volume = msk_volume.permute(0,3,1,2) # 1*d*w*h
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_volume_target = torch.zeros(1,d,3,self.args.image_size,self.args.image_size) #1*d*3*1024*1024 (each pixel has three channel)
        msk_volume_prompt_target = torch.zeros(1,d,self.args.image_size,self.args.image_size) 
        # generate mask and prompt
        if self.if_prompt:
            pc_list = []
            pl_list = []
            if self.prompt_type =='point':
                for slice_idx in range(img_volume.shape[1]):
                    img_3c = einops.repeat(img_volume[0,slice_idx][None,:,:], '1 w h -> k w h', k=3) # 1*3*w*h
                    img_3c_norm = transforms.Resize((self.args.image_size,self.args.image_size))(normalize(img_3c))
                    img_volume_target[0,slice_idx] = img_3c_norm[0]
                    
                    mask_slice = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk_volume[0,slice_idx][None,:,:])
                    mask_cls = np.array(mask_slice[0],dtype=int)
                    prompt,mask_now = get_first_prompt(mask_cls,region_type=self.region_type,prompt_num=self.prompt_num)
                    pc_list.append(torch.as_tensor(prompt[:,:2], dtype=torch.float)[None,:,:]) # 5*2
                    pl_list.append(torch.as_tensor(prompt[:, -1], dtype=torch.float)[None,:].reshape(1,-1)) # 5*1
                    msk_volume_prompt_target[0,slice_idx] = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                pcs = torch.cat(pc_list,dim=0)
                pls = torch.cat(pl_list,dim=0)
                
                return {'image_input':img_volume_target,
                        'image_ori':img_volume,
                    'mask':msk_volume_prompt_target,
                    'mask_ori':msk_volume,
                    'point_coords': pcs,
                    'point_labels':pls,
                    'img_name':img_path,
                    'atten_map':attention_map,
            }
            elif self.prompt_type =='box':
                prompt,mask_now = get_top_boxes(mask_cls,region_type=self.region_type)
                box = torch.as_tensor(prompt, dtype=torch.float)
                msk = torch.unsqueeze(torch.tensor(mask_now,dtype=torch.long),0)
                return {'image':img,
                    'mask':msk,
                    'boxes':box,
                    'img_name':img_path,
            }
        else:
            #print(img_volume.max())
            for slice_idx in range(img_volume.shape[1]):
                img_3c = einops.repeat(img_volume[0,slice_idx][None,:,:], '1 w h -> k w h', k=3) # 1*3*w*h
                img_3c_norm = transforms.Resize((self.args.image_size,self.args.image_size))(normalize(img_3c))
                img_volume_target[0,slice_idx] = img_3c_norm[0]
                mask_slice = transforms.Resize((self.args.image_size,self.args.image_size),InterpolationMode.NEAREST)(msk_volume[0,slice_idx][None,:,:])
                msk_volume_prompt_target[0,slice_idx] = mask_slice.clone().detach().long()
            return {'image_input':torch.squeeze(img_volume_target),
                'image_ori':img_volume,
                'mask':torch.squeeze(msk_volume_prompt_target),
                'img_name':img_path,
                'atten_map':attention_map,
        }
        
        
        
class MRI_dataset_3D_auto(Dataset):
    '''
    It is a dataset for reading 3D volumes
    if choose if_prompt = False: it is for automatically segmenation
        image_input: shape is 1*d*3*w*h: d is depth of volume, and 3 means 3 channel for each slice.
    '''
    def __init__(self,args, img_folder, mask_folder, img_list,phase='train',channel_num=1,crop=False,crop_size=1024,targets=['femur','hip'],part_list=['all'],cls=1,if_prompt=True,prompt_type='point',region_type='largest_10',prompt_num=10,if_attention_map=None):
        super(MRI_dataset_3D_auto, self).__init__()
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.crop = crop
        self.crop_size = crop_size
        self.phase = phase
        self.channel_num=channel_num
        self.targets = targets
        self.segment_names_to_labels = []
        self.args = args
        self.cls = cls
        self.if_prompt = if_prompt
        self.region_type = region_type
        self.prompt_type = prompt_type
        self.prompt_num = prompt_num
        self.depth = self.args.depth
        self.if_attention_map = if_attention_map
        
        for i,tag in enumerate(targets):
            self.segment_names_to_labels.append((tag,i))
            
        namefiles = open(img_list,'r')
        self.data_list = namefiles.read().split('\n')[:-1]

        aug_transforms = {
                        tio.transforms.RandomElasticDeformation():0.2,
                        tio.transforms.RandomAffine():0.3,
                        tio.transforms.RandomNoise():0.1,
                         }
        train_transforms = tio.Compose([tio.OneOf(aug_transforms, p = 0.995),
                             tio.RescaleIntensity(out_min_max=(0, 1)),
                            ])
        eval_transforms = tio.RescaleIntensity(out_min_max=(0, 1))
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms
            
    def __len__(self):
        return len(self.data_list)
    
    def _augment_data(self,data):
        if self.phase == 'train':
            data = self.train_transforms(data)
        else:
            data = self.eval_transforms(data)
        return data
        
    def __getitem__(self,index):
        # load image and the mask
        data = self.data_list[index]
        img_path = data.split(' ')[0]
        mask_path = data.split(' ')[1]
        #print(img_path,mask_path)
        state = torch.get_rng_state()
        
        img = nib.load(os.path.join(self.img_folder,img_path))
        img_array = np.expand_dims(np.array(img.dataobj),axis=0) #1*w*h*d
        img_array = (img_array-img_array.min())/(img_array.max()-img_array.min()+1e-8)

        mask = nib.load(os.path.join(self.mask_folder,mask_path)) # 1*w*h*d
        msk= np.expand_dims(np.array(mask.dataobj),axis=0)
        
        if 'all' in self.targets: # combine all targets as single target
            msk = np.array(np.array(msk,dtype=int)>0,dtype=int)
        else:
            msk = np.array(msk,dtype=int)
            
        #extract the target objects
        mask_array = np.array(msk==self.cls,dtype=int)
        
        #print(os.path.join(self.img_folder,img_path),img_array.shape,mask_array.shape)
        obj = tio.Subject(
                    image = tio.ScalarImage(tensor = img_array),
                    mask = tio.LabelMap(tensor = mask_array)
                )
        
        aug_obj = self._augment_data(obj)
        resize = tio.Resize((self.args.depth,self.args.image_size,self.args.image_size),image_interpolation='nearest')
        aug_obj = resize(aug_obj)
        
        img_volume = aug_obj.image[tio.DATA].float()  # 1*w*h*d
        #print(img_volume.max())
        msk_volume = aug_obj.mask[tio.DATA].float() 
        

        return {'image_input':torch.squeeze(img_volume),
            'image_ori':img_volume,
            'mask':torch.squeeze(msk_volume ),
            'img_name':img_path,
        }

#from segment_anything import SamPredictor, sam_model_registry
from models.sam import SamPredictor, sam_model_registry
from models.sam.modeling.prompt_encoder import auto_cls_emb
from models.sam.utils.transforms import ResizeLongestSide
#Scientific computing 
import numpy as np

import os
#Pytorch packages
import torch
from torch import nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
#Visulization
from torchvision import transforms
#Others
from torch.utils.data import DataLoader
from dataset_bone import MRI_dataset_multicls
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from losses import DiceLoss
from dsc import dice_coeff
import monai

import cfg
args = cfg.parse_args()
args.if_mask_decoder_adapter=True
args.if_encoder_adapter = True
args.if_warmup = True
args.lr = 1e-4
args.decoder_adapt_depth=2
args.encoder_adapter_depths=[0,1,10,11]
args.pretrain_weight = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_data/588/fine-tune-sam/Medical-SAM-Adapter/mobile_sam.pt'

epochs = 1000

def run_one_epoch(dataloader, model, optimizer=None, criterion1=None, criterion2=None, args=None, iter_num=None,\
                  writer=None, epoch=None, max_iter=-1, train=True, moe_print=False, sub=False):
    

    total_loss = 0
    if train:
        #model.train() 
        model.eval()
    else:
        model.eval()
        dsc = 0

    if train and 0:
        if sub:
            lower, upper = 0, 5
        else:
            lower, upper = 5, 10
        
        print('Updating adapter', lower, upper)

        def check_name(name, lower, upper):
            accepted_format = []
            for i in range(lower, upper):
                accepted_format.append('adapter.%s.' % i)
                accepted_format.append('adapter2.%s.' % i)

            # Always append 0 when loc specific
            accepted_format.append('adapter.%s.' % 0)
            accepted_format.append('adapter2.%s.' % 0)
            
            ret = False
            for af in accepted_format:
                if af in name.lower():
                    ret = True
            return ret

        for n, value in model.named_parameters():
            if check_name(n, lower, upper) or 'gater' in n.lower() or 'noise' in n.lower():
                value.requires_grad = True
            else:
                value.requires_grad = False


    for i,data in enumerate(dataloader):
        imgs = data['image'].cuda()
        
        if moe_print and (i == 0 or i == 400):
            img_emb, moe_loss_en, gates_total = model.image_encoder(imgs, training=train, moe_print=True)
        else:
            img_emb, moe_loss_en, gates_total = model.image_encoder(imgs, training=train)

        # automatic masks contaning all muscles
        msks = torchvision.transforms.Resize((args.out_size,args.out_size))(data['mask_cls'])
        msks = msks.cuda()

        sparse_emb, dense_emb = model.prompt_encoder(
                                    points=None,
                                    boxes=None,
                                    masks=None,
                                )
                
        pred, _, moe_loss_de, gates_total_de = model.mask_decoder(
                                                    image_embeddings=img_emb,
                                                    image_pe=model.prompt_encoder.get_dense_pe(), 
                                                    sparse_prompt_embeddings=sparse_emb,
                                                    dense_prompt_embeddings=dense_emb, 
                                                    multimask_output=True,
                                                    training=train,
                                               )
        
        loss_dice = criterion1(pred,msks.float()) 
        loss_ce = criterion2(pred,torch.squeeze(msks.long(),1))
        loss =  loss_dice + loss_ce
        
        if moe_loss_en > 0:
            loss += moe_loss_en 
        if moe_loss_de > 0:
            loss += moe_loss_de
       
        total_loss += loss.item()

        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            else:
                if args.if_warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.lr * (1.0 - shift_iter / max_iter) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_

            iter_num+=1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

        else:
            dsc += dice_coeff((pred[:,1,:,:].cpu()>0).long(),msks.cpu().long()).item()

    total_loss /= (i+1)
    if train:
        print('Epoch num {}| train loss {}'.format(epoch,total_loss))
        return iter_num 
    else:
        dsc /= (i+1)
        writer.add_scalar('eval/loss', total_loss, epoch)
        writer.add_scalar('eval/dice', dsc, epoch)
        return total_loss, dsc



def train_model(loaders,dir_checkpoint,epochs, subloaders=None, max_iter=-1, moe=0, k=0):
    trainloader = loaders[0]
    valloader   = loaders[1]
    testloader  = loaders[2]
        
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr
    
    writer = SummaryWriter(dir_checkpoint + '/log')
    
    sam = sam_model_registry["vit_t"](args,checkpoint=args.pretrain_weight,num_classes=3, moe=moe, k=k)
    for n, value in sam.named_parameters():
        value.requires_grad = False
    for n, value in sam.named_parameters():
        if "adapter" in n.lower() or 'gater' in n.lower() or 'noise' in n.lower():
            value.requires_grad = True
        if 'centroid' in n.lower():
            value.requires_grad = False
    sam.to('cuda')
    

    optimizer = optim.AdamW(sam.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #learning rate decay

    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True,reduction='mean')
    criterion2 = nn.CrossEntropyLoss()

    #pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    val_largest_single = 0
    test_dsc = 0
    test_dsc_single = 0
    last_update_epoch = 0

    iter_num = 0

    for epoch in range(epochs):
        iter_num = run_one_epoch(trainloader, sam, optimizer, criterion1, criterion2, args, iter_num, writer, epoch, max_iter)

        if epoch % 5 == 0:
            with torch.no_grad():
                eval_loss, dsc = run_one_epoch(valloader, sam, None, criterion1, criterion2, args, writer=writer, train=False, moe_print=True)
                print('Eval Epoch num {} | val loss {} | dsc {} '.format(epoch,eval_loss,dsc))
                
                if dsc > val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    Path(dir_checkpoint).mkdir(parents=True,exist_ok=True)
                    _, test_dsc = run_one_epoch(testloader, sam, None, criterion1, criterion2, args, writer=writer, train=False)
                    print('saving to', dir_checkpoint + '/checkpoint_best_byall.pth')
                    torch.save(sam.state_dict(), dir_checkpoint + '/checkpoint_best_byall.pth')

                if (epoch-last_update_epoch) >= 100:
                    print('Training finished###########')
                    break
                #print('largest DSC now: {} / {}'.format(val_largest_dsc, val_largest_single))
                #print('largest Test DSC now: {} / {}'.format(test_dsc, test_dsc_single))
                print('largest DSC now: {} / {}'.format(val_largest_dsc, test_dsc))

    writer.close()                  
                        
if __name__ == "__main__":
    date = '1115_final'

    vol_type = 'all'

    dataset_name = 'PrivateMuscleonly_%sall' % date
    img_folder =  '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/private_data/2D-slices/images'
    mask_folder = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/private_data/2D-slices/masks'
    

    moe = 5
    k = 5

    controlled = False

    print(moe, k, controlled)

    test_img_list =  '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_all_val_2dslices.txt' % date
    train_img_list_addition = None

    if 1:
        train_img_list = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset1_2dslices.txt' % (date, vol_type)
        val_img_list =   '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset2_2dslices.txt' % (date, vol_type)

        train_img_list = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_2dslices.txt' % (date, vol_type)
        val_img_list =   '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_val_2dslices.txt' % (date, vol_type)

        dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_%sC%s' % (moe, k)

        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_traingate_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_pretrained_gateonly_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_pretrained_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_perloc_bs1_%sC%s' % (moe, k)

        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_perloc_labelmask_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_perloc_labelmask_nocommon_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_perloc_labelmask_nocommon_gateexplicit_%sC%s' % (moe, k)

        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_loadpretrained_loc_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_loadpretrained_seq_%sC%s' % (moe, k)

        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_lowrank_%sC%s' % (moe, k)


    if 1:
        version = 'exclude'

        train_img_list = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset1_paired_%s_2dslices.txt' % (date, vol_type, version)
        val_img_list =   '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset2_2dslices.txt' % (date, vol_type)
        dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_%s_trainonly_%sC%s' % (version, moe, k)

        train_img_list = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_paired_exclude_2dslices.txt' % (date, vol_type)
        val_img_list =   '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_val_2dslices.txt' % (date, vol_type)

        #train_img_list = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_paired_exclude_2dslices_noresize.txt' % (date, vol_type)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_%s_trainonly_noresize_%sC%s' % (version, moe, k)

        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_trainonly_labelspecficexpert_%sC%s' % (moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_%s_trainonly_loadpretrained_loc_%sC%s' % (version, moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_%s_trainonly_loadpretrained_paired_loc_%sC%s' % (version, moe, k)
        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_%s_trainonly_loadpretrained_all_loc_%sC%s' % (version, moe, k)

        #dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_trainonly_loadpretrained_seq_%sC%s' % (moe, k)

        #train_img_list_addition = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset1_2dslices.txt' % (date, vol_type)

    if 0:
        version = 'exclude'
        train_img_list = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset1_paired_%s_2dslices.txt' % (date, vol_type, version)
        val_img_list =   '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/img_lists/private_hip_%s_%s_train_subset2_paired_%s_2dslices.txt' % (date, vol_type, version)
        dir_checkpoint = 'model_checkpoints_hd/2D-MobileSAM-encoderdecoder-adapter_'+dataset_name+'_moe_paired_%s_trainval_%sC%s' % (version, moe, k)
    
    print(dir_checkpoint)

    bs = 8
    num_workers = 6
    label_mapping = '/mnt/largeDrives/sevenTBTwo/bone_proj/codes_for_muscle/segment_names_to_labels.pickle'

    train_dataset = MRI_dataset_multicls(args,img_folder, mask_folder, train_img_list,phase='train',targets=['muscle'],delete_empty_masks=False,label_mapping=label_mapping)
    val_dataset   = MRI_dataset_multicls(args,img_folder, mask_folder, val_img_list,phase='val',targets=['muscle'],delete_empty_masks=False,label_mapping=label_mapping)
    test_dataset  = MRI_dataset_multicls(args,img_folder, mask_folder, test_img_list,phase='val',targets=['muscle'],delete_empty_masks=False,label_mapping=label_mapping)
    
    trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valloader   = DataLoader(val_dataset,  batch_size=1, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    max_iter = epochs * len(trainloader)
    
    if train_img_list_addition is not None:
        train_sub_dataset = MRI_dataset_multicls(args,img_folder, mask_folder, train_img_list_addition,phase='train',targets=['muscle'],delete_empty_masks=False,label_mapping=label_mapping)
        trainsubloader = DataLoader(train_sub_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    else:
        trainsubloader = None

    train_model([trainloader,valloader,testloader], dir_checkpoint,epochs, \
                subloaders=[trainsubloader],max_iter=max_iter, moe=moe, k=k)

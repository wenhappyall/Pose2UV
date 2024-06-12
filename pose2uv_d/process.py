import torch
import torch.optim as optim
from Scheduler import GradualWarmupScheduler
import sys
from torch.autograd import Variable
from utils.imutils import *
import numpy as np
import os
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from alphapose_module.alphapose_core import AlphaPose_Predictor
from alphapose_module.alphapose.utils.metrics import evaluate_mAP
import json



# 2D pose estimation
alpha_config = R'D:\Program Files (x86)\Pose2UV-main\alphapose_module\configs\halpe_26\256x192_res50_lr1e-3_1x.yaml'
alpha_checkpoint = R'data/halpe26_fast_res50_256x192.pth'
alpha_thres = 0.1
alpha_predictor = AlphaPose_Predictor(alpha_config, alpha_checkpoint, alpha_thres)

modelConfig = {
        "state": "train", # or eval
        "epoch": 200,
        "batch_size": 80,
        "T": 1000, # total step for model training
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4, # noise figure at the beginning
        "beta_T": 0.02, # noise figure at the end
        "img_size": 32, 
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        "my_t": 50
        }


def to_device(data, device):
    temp = {}
    if 'mask' in data.keys(): # check if the key "mask" exists in the data dictionary
        temp['mask'] = [item.to(device) for item in data['mask']] # iterate each item in the data['mask']list and call item to move it to the specified device, place the moved element in the temp['mask']list
    if 'fullheat' in data.keys():
        temp['fullheat'] = [item.to(device) for item in data['fullheat']]
    if 'partialheat' in data.keys():
        temp['partialheat'] = [item.to(device) for item in data['partialheat']]
    if 'img_path' in data.keys():
        temp['img_path'] = data['img_path']
    if 'fullheatmap' in data.keys():
        temp['fullheatmap'] = [item.to(device) for item in data['fullheatmap']]
    # iterate over the key/value pairs(k,v) of the data dictionary, If the key (k) is not in the list
    # Move v to the device and convert it to float(). add the key/value to a new dictionnary named data.,
    data = {k:v.to(device).float() for k, v in data.items() if k not in ['mask', 'fullheat', 'partialheat', 'img_path', 'fullheatmap']}

    # dictionary union operator (**) to merge the two dictionaries and overwrite the temp key-value pairs in the data dictionary.
    data = {**temp, **data}

    return data  

def viz_poseseg(pred_hm=None, gt_hm=None, pred_ms=None, gt_ms=None, img=None):
    pred_hm = pred_hm.detach().data.cpu().numpy().astype(np.float32)
    gt_hm = gt_hm.detach().data.cpu().numpy().astype(np.float32)
    pred_ms = pred_ms.detach().data.cpu().numpy().astype(np.float32)
    gt_ms = gt_ms.detach().data.cpu().numpy().astype(np.float32)
    img = img.detach().data.cpu().numpy().astype(np.float32)
    for phm, ghm, pms, gms, im in zip(pred_hm, gt_hm, pred_ms, gt_ms, img):
        im = im.transpose((1,2,0)) 
        pms = pms[0]  
        gms = gms[0]  

        for p_kp, g_kp in zip(phm, ghm):
            scale = p_kp.shape[0] / g_kp.shape[0] 
            g_kp = cv2.resize(g_kp, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # if p_kp.max() > 0.3:
            #     p_kp = np.mean(np.where(p_kp == np.max(p_kp)), axis=1).astype(np.int64)
            #     im = cv2.circle(im, (p_kp[1], p_kp[0]), 2, (0,0,255),-1)

            # if g_kp.max() > 0.3:
            #     g_kp = cv2.resize(g_kp, (256,256),interpolation=cv2.INTER_CUBIC)
            #     g_kp = np.mean(np.where(g_kp == np.max(g_kp)), axis=1).astype(np.int64)
            #     im = cv2.circle(im, (g_kp[1], g_kp[0]), 2, (0,255,0),-1)

        cv2.imshow("img", im)
        # cv2.imshow("p_mask",pms/255)
        cv2.imshow("g_mask",gms)
        cv2.waitKey()


def viz_masks(m0, m1, m2, m3, mask):
    m_0 = m0.detach().data.cpu().numpy().astype(np.float32)
    m_1 = m1.detach().data.cpu().numpy().astype(np.float32)
    m_2 = m2.detach().data.cpu().numpy().astype(np.float32)
    m_3 = m3.detach().data.cpu().numpy().astype(np.float32)
    mask_viz = mask.detach().data.cpu().numpy().astype(np.float32)
    for m0, m1, m2, m3, mask in zip(m_0, m_1, m_2, m_3, mask_viz):

        m0 = m0.transpose(1,2,0)
        m1 = m1.transpose(1,2,0)
        m2 = m2.transpose(1,2,0)
        m3 = m3.transpose(1,2,0)
        mask = mask.transpose(1,2,0)

        cv2.imshow("m0",m0)
        cv2.imshow("m1",m1)
        cv2.imshow("m2",m2)
        cv2.imshow("m3",m3)
        cv2.imshow("mask",mask)
        cv2.waitKey()

def poseseg_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10) 
    len_data = len(train_loader)    # calculate the length of train_dateset, the number of sample
    model.model.train(mode=True)    # set the model mode to train mode
    train_loss = 0.                 # initial training loss as 0, accumulate the loss value through the training process
    
    # diffusion optimizer
    optimizer = torch.optim.AdamW(
    model.model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
    optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    for i, data in enumerate(train_loader): # get data from def __getitem__(): self.create_data(): create_poseseg()

        # batchsize = data['img'].size(0)  
    
        optimizer.zero_grad()
        data = to_device(data, device)  # move the data to the specified computing device(CPU or GPU)

        pred = model.model(data)
        
        loss = pred['loss'].sum() / 1000
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.model.parameters(), modelConfig["grad_clip"]) # avoid 
        optimizer.step()
        # loss, loss_dict = loss_func.calcul_trainloss(pred, data)

        # # backward
        # model.optimizer.zero_grad()
        # loss.backward()

        # # optimize
        # model.optimizer.step()
        # loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss))
        if i % 10000 == 0: 
            torch.save(model.model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(i) + "_dataloader.pt"))   
        # train_loss += loss_batch
    warmUpScheduler.step()
    torch.save(model.model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(epoch) + "_.pt"))   
    # return train_loss/len_data

lsp14_to_lsp13 = [0,1,2,3,4,5,6,7,8,9,10,11,13]
halpe_to_lsp = [16,14,12,11,13,15,10,8,6,5,7,9,18,0]
alphapose_loss = []
alphapose_evaluation = []
def calcu_alphapose(img_path, bbox, gt):
    for img, box, gt_kp in zip(img_path, bbox, gt.detach().cpu().numpy()):
        img = cv2.imread(img)
        pred_2d = alpha_predictor.predict(img, box.detach().cpu().numpy())
        print(pred_2d)
        pred_2d = pred_2d[0][halpe_to_lsp][lsp14_to_lsp13]
        print(pred_2d)      
        conf = gt_kp[:,-1]
        pred_2d = pred_2d[conf>0]
        gt_kp = gt_kp[conf>0]

        mpjpe2d = np.sqrt(np.sum((pred_2d[:,:2] - gt_kp[:,:2])**2, axis=-1)).mean() 
        alphapose_loss.append(mpjpe2d)
                
       # get pred_2d json
        with open("pred_2d.json", "w") as f:
            json.dump(pred_2d, f) 
        # evaluate 
        evaluation = evaluate_mAP(res_file='pred_2d.json', ann_type='keypoints', ann_file='D:\Program Files (x86)\Pose2UV-main\data\person_keypoints_val2017.json', silence=True, halpe=False)
        alphapose_evaluation.append(evaluation)

def poseseg_test(model, loss_func, loader,epoch,  viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    kpt_json = []
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            # batchsize = data['img'].size(0)
            data = to_device(data, device)
            
            # alphapose
            # calcu_alphapose(data['img_path'], data['bbox'], data['gt_kp2d'])
            
            # forward
            pred = model.model(data)
            # preheat = pred['preheat'][4] # shape = [15,17,256,256]
            # fullheatmap = data['fullheatmap'][-1]
            
            # diffusion loss
            x_0 = pred['x_0'][0]
            x_0 = x_0.detach().cpu().numpy().astype(np.float32)
            x_0 = x_0.transpose(1, 2, 0)
            x_0 = np.max(x_0, axis=2)
            x_0 = convert_color(x_0*255)
            heatmap_name = "%05ddiffusion_pred_heatmap.jpg" %(epoch)
            if not os.path.exists(modelConfig["sampled_dir"]):
                os.makedirs(modelConfig["sampled_dir"])
            cv2.imwrite(os.path.join(modelConfig["sampled_dir"], heatmap_name), x_0)
            # calculate loss_poseseg
            # loss, loss_dict = loss_func.calcul_testloss(pred, data)
            
            # save json_poseseg     
            # for j in range(preheat.shape[0]): # num=15
            #     heatmap = preheat[j] 
            #     hm_size = [heatmap.shape[1], heatmap.shape[2]]
            #     bbox = data['bbox'][j].tolist()
            #     img = cv2.imread(data['img_path'][j])
            #     offset=data['offset'][j].cpu().numpy()
            #     scale=data['scale'][j].cpu().numpy()
            #     pose_coords, pose_scores = heatmap_to_coord(
            #         heatmap, bbox, hm_shape=hm_size, norm_type=None, hms_flip = None, offset=offset,scale=scale, )
            #     keypoints = np.concatenate((pose_coords, pose_scores), axis=1)
            #     keypoints = keypoints.reshape(-1).tolist()        
                
            #     data_json = dict()
            #     data_json['bbox'] = bbox
            #     data_json['image_id'] = int(data['img_id'][j])
            #     data_json['score'] = float(np.mean(pose_scores) + 1.25 * np.max(pose_scores))
            #     data_json['category_id'] = 1
            #     data_json['keypoints'] = keypoints
            #     # img2 = draw_pose(img, pose_coords, (0,0,255))
            #     # vis_img("img", img2)
                
            #     kpt_json.append(data_json)
                        
            # save results _diffusion
            
            # save results_poseseg
            # if i < 1:
            #     results = {}
            #     results.update(imgs=data['img'].detach().cpu().numpy().astype(np.float32))
            #     results.update(pred_heats=pred['preheat'][-1].detach().cpu().numpy().astype(np.float32))
            #     # results.update(pred_masks=pred['premask'][-1].detach().cpu().numpy().astype(np.float32))
            #     results.update(gt_heats=data['partialheat'][-1].detach().cpu().numpy().astype(np.float32))
            #     results.update(gt_masks=data['mask'][-1].detach().cpu().numpy().astype(np.float32))
            #     model.save_poseseg_results(results, i, batchsize)
                
        #     loss_batch = loss.detach()
        #     print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch))
        #     # print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), loss_dict)
        #     loss_all += loss_batch
        # sysout = sys.stdout  
        # loss_all = loss_all / len(loader)
        # # # evaluate_mAP  
        # with open('data/validate_gt_kpt.json', 'w') as fid:
        #     json.dump(kpt_json, fid)
        # res = evaluate_mAP('data/validate_gt_kpt.json', ann_type='keypoints', ann_file='data/person_keypoints_val2017.json', halpe=None)
        # sys.stdout = sysout
        # mAP = res
           
        # return loss_all, mAP
        

def get_max_pred(heatmaps):
    num_joints = heatmaps.shape[0] # heatmap.shape : (17, 64, 48)
    width = heatmaps.shape[2] # 
    heatmaps_reshaped = heatmaps.reshape((num_joints, -1)) # reshape to Two-dimensional tensor, -1:H*W
    idx = np.argmax(heatmaps_reshaped, 1) # index：(max probability) from second dim
    maxvals = np.max(heatmaps_reshaped, 1) # value: max confidence from second dim

    maxvals = maxvals.reshape((num_joints, 1)) # reshape to two-dimensional tensor by num_joint
    idx = idx.reshape((num_joints, 1)) # reshape to two-dimensional tensor by num_joint

    preds = np.tile(idx, (1, 2)).astype(np.float32) # create two-dimensional arry (keypoint coordinates) by duplicating idx

    preds[:, 0] = (preds[:, 0]) % width # caculate x coordinate 
    preds[:, 1] = np.floor((preds[:, 1]) / width) # caculate y coordinate

    # create mask to filters out keypoint predictions with low confidence
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 2)) # compare maxvals with (0.0), > True/ < False and create two-dimensional arry by duplicating the result
    pred_mask = pred_mask.astype(np.float32) # convert boolean to float

    preds *= pred_mask
    return preds, maxvals

def heatmap_to_coord(hms, bbox, hms_flip=None, offset=None, scale=None, **kwargs):
    if hms_flip is not None:
        hms = (hms + hms_flip) / 2
    if not isinstance(hms,np.ndarray): # tensor to numpy
        hms = hms.cpu().data.numpy()
    coords, maxvals = get_max_pred(hms) 
        
    hm_h = hms.shape[1]
    hm_w = hms.shape[2]

    # post-processing: fine-tune the prediction coordinates to the direction of neighboring pixels with higher heatmap values
    for p in range(coords.shape[0]):
        hm = hms[p] # joint p's heatmap 
        px = int(round(float(coords[p][0]))) # coordinate x of joint p 
        py = int(round(float(coords[p][1]))) # coordinate y of joint p 
        if 1 < px < hm_w - 1 and 1 < py < hm_h - 1: # check coords are within the valid range
            diff = np.array((hm[py][px + 1] - hm[py][px - 1], 
                             hm[py + 1][px] - hm[py - 1][px])) # diff[0]:difference in the x direction diff[1]:difference in the y direction
            coords[p] += np.sign(diff) * .25 #sign: (+1/0/-1) ，0.25是 scale


    coords[:, :2] = coords[:, :2] - offset
    coords[:, :2] = coords[:, :2] / scale
    preds = coords
    return preds, maxvals
    # preds = np.zeros_like(coords)

    # # transform bbox to scale
    # xmin, ymin, xmax, ymax = bbox
    # w = xmax - xmin
    # h = ymax - ymin
    # center = np.array([xmin + w * 0.5, ymin + h * 0.5]) # calculate the normalization factor
    # scale = np.array([w, h])
    # # Transform back / normalization
    # for i in range(coords.shape[0]):
    #     preds[i] = transform_preds(coords[i], center, scale,
    #                                [hm_w, hm_h])

    # return preds, maxvals

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def posenet_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        if torch.cuda.is_available():
            hmgt = [Variable(item).to(device) for item in data['heatmaps']]
            img = Variable(data['img']).to(device)
            crop = Variable(data['crop']).to(device)
        else:
            print('CUDA error')
            sys.exit(0)

        # forward
        output = model.model(crop)

        # calculate loss
        loss = loss_func.calcul_heatmaploss(output, hmgt)
        # visualize
        if viz:
            # viz_poseseg(pred_hm=output[3], gt_hm=hmgt[2], pred_ms=output[9][:,14,:,:], gt_ms=data['mask'], img=img)

            test = output[3].detach().cpu().numpy().astype(np.float32)
            test_img = img.detach().cpu().numpy().astype(np.float32)
            gt = hmgt[2].detach().cpu().numpy().astype(np.float32)
            test_img = test_img[0].transpose((1,2,0))
            vis_img("img", test_img)
            for t in range(14):
                temp = convert_color(test[0][t]*255)
                gtt = convert_color(gt[0][t]*255)
                vis_img("hm", temp)
                vis_img("gt", gtt)

        # backward
        model.optimizer.zero_grad()
        loss.backward()
        # optimize
        model.optimizer.step()

        loss_batch = loss.detach() / batchsize
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch))
        train_loss += loss_batch

    return train_loss/len_data

def posenet_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):

    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                hmgt = [Variable(item).to(device) for item in data['heatmaps']]
                img = Variable(data['img']).to(device)
                crop = Variable(data['crop']).to(device)
            else:
                print('CUDA error')
                sys.exit(0)

            # forward
            output = model.model(crop)

            # calculate loss
            loss = loss_func.calcul_heatmaploss(output, hmgt)
            # visualize
            if viz:
                # viz_poseseg(pred_hm=output[8], gt_hm=hmgt[2], pred_ms=output[9][:,14,:,:], gt_ms=data['mask'], img=img)

                test = output[3].detach().cpu().numpy().astype(np.float32)
                test_img = crop.detach().cpu().numpy().astype(np.float32)
                gt = hmgt[2].detach().cpu().numpy().astype(np.float32)
                test_img = test_img[0].transpose((1,2,0))
                vis_img("img", test_img)
                for t in range(14):
                    temp = convert_color(test[0][t]*255)
                    gtt = convert_color(gt[0][t]*255)
                    vis_img("hm", temp)
                    vis_img("gt", gtt)

                #viz_masks(m0, m1, m2, m3, mask, mask1)
            # save results
            if i < 0:
                results = {}
                results.update(img=img.detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)
                
            loss_batch = loss.detach() / batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch))
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def segnet_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        if torch.cuda.is_available():
            msgt = [Variable(item).to(device) for item in data['masks']]
            full_hm = [Variable(item).to(device) for item in data['full_heatmaps']]
            img = Variable(data['img']).to(device)
            # oc_index = Variable(data['oc_index']).to(device)
        else:
            print('CUDA error')
            sys.exit(0)

        # forward
        output = model.model(img, full_hm) #img,crop

        # calculate loss
        loss = loss_func.calcul_segloss(output, msgt)

        # backward
        model.optimizer.zero_grad()
        loss.backward()
        # optimize
        model.optimizer.step()
        loss_batch = loss.detach() / batchsize
        print('epoch：%d/%d, batch：%d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch))
        train_loss += loss_batch
    return train_loss/len_data

def segnet_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                msgt = [Variable(item).to(device) for item in data['masks']]
                full_hm = [Variable(item).to(device) for item in data['full_heatmaps']]
                img = Variable(data['img']).to(device)
            else:
                print('CUDA error')
                sys.exit(0)

            # forward
            output = model.model(img, full_hm)

            # calculate loss
            loss = loss_func.calcul_segloss(output, msgt)

            # save results
            if i < 5:
                results = {}
                results.update(img=img.detach().cpu().numpy().astype(np.float32))
                results.update(pre_mask=output[4].detach().cpu().numpy().astype(np.float32))
                results.update(gt_mask=data['mask'].detach().cpu().numpy().astype(np.float32))
                model.save_seg(results, i, batchsize)
            loss_batch = loss.detach() / batchsize
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch))
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all


def segnet_uv_vae_train(model, loss_func, train_loader, epoch, num_epoch,\
                        viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model training' + '-' * 10)
    len_data = len(train_loader)
    model.model.train(mode=True)
    train_loss = 0.
    for i, data in enumerate(train_loader):
        batchsize = data['img'].size(0)
        data = to_device(data, device)

        output = model.model(data)

        loss, loss_dict = loss_func.calcul_trainloss(output, data)

        # visualize
        if viz:
            model.viz_input(input_ht=data['fullheat'][-1], output_ht=output['heatmap'], rgb_img=data['img'], pred=output['pred_uv'], mask=output['pred_mask'][-1])
        # backward
        model.optimizer.zero_grad()
        loss.backward()

        # optimize
        model.optimizer.step()
        loss_batch = loss.detach()
        print('epoch: %d/%d, batch: %d/%d, loss: %.6f' %(epoch, num_epoch, i, len_data, loss_batch), loss_dict)
        train_loss += loss_batch
    return train_loss/len_data

def segnet_uv_vae_test(model, loss_func, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            batchsize = data['img'].size(0)
            data = to_device(data, device)

            # forward
            output = model.model(data)
            
            # calculate loss
            loss, loss_dict = loss_func.calcul_testloss(output, data)

            # save results
            if i < 4:
                results = {}
                results.update(mask=output['mask'].detach().cpu().numpy().astype(np.float32))
                results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
                results.update(uv=output['pred_uv'].detach().cpu().numpy().astype(np.float32))
                results.update(uv_gt=data['gt_uv'].detach().cpu().numpy().astype(np.float32))
                results.update(rgb_img=data['img'].detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)
                
            loss_batch = loss.detach()
            print('batch: %d/%d, loss: %.6f ' %(i, len(loader), loss_batch), loss_dict)
            loss_all += loss_batch
        loss_all = loss_all / len(loader)
        return loss_all

def demo(model, yolox_predictor, alpha_predictor, loader, device=torch.device('cpu')):
    print('-' * 10 + 'model testing' + '-' * 10)
    loss_all = 0.
    model.model.eval()
    with torch.no_grad():
        for i, img_path in tqdm(enumerate(loader), total=len(loader)):
            img = cv2.imread(img_path)

            det, _ = yolox_predictor.predict(img_path, viz=False)
            poses = alpha_predictor.predict(img, det['bbox'])

            # alpha_predictor.visualize(img, poses, viz=False)

            data = loader.prepare(img, det['bbox'], poses, device)

            # forward
            output = model.model(data)

            # save results
            results = {}
            results.update(scales=data['scale'].astype(np.float32))
            results.update(offsets=data['offset'].astype(np.float32))
            results.update(pred_heats=output['preheat'][-1].detach().cpu().numpy().astype(np.float32))
            results.update(pred_masks=output['premask'][-1].detach().cpu().numpy().astype(np.float32))
            # results.update(uv=output['pred_uv'].detach().cpu().numpy().astype(np.float32))
            results.update(img=data['img'])
            model.save_pose( results, iter)
            # model.save_demo_results(results, i, img_path)
                

def EvalModel(model, evaltool, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'evaluation' + '-' * 10)
    abs_errors, errors, error_pas, abs_pcks, pcks, imnames, joints, joints_2ds, vertex_errors = [], [], [], [], [], [], [], [], []
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                rgb_img = Variable(data['img']).to(device)
                full_hm = [Variable(item).to(device) for item in data['fullheat']]
            else:
                print('CUDA error')
                sys.exit(0)
            
            # forward
            output = model.model(rgb_img, full_hm)
            abs_error, error, error_pa, abs_pck, pck, imname, joint, joints_2d, vertex_error = evaltool(output, data)
            abs_errors += abs_error
            errors += error
            error_pas += error_pa
            abs_pcks += abs_pck
            pcks += pck
            imnames += imname
            joints += joint
            joints_2ds += joints_2d
            vertex_errors += vertex_error

            # # save results
            # if i < 4:
            #     results = {}
            #     results.update(mask=output['mask'].detach().cpu().numpy().astype(np.float32))
            #     results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
            #     results.update(pred=output['decoded'].detach().cpu().numpy().astype(np.float32))
            #     results.update(uv_gt=uv_gt.detach().cpu().numpy().astype(np.float32))
            #     results.update(rgb_img=rgb_img.detach().cpu().numpy().astype(np.float32))
            #     model.save_results(results, i, batchsize)
        
        abs_error = np.mean(np.array(abs_errors))
        error = np.mean(np.array(errors))
        error_pa = np.mean(np.array(error_pas))
        abs_pck = np.mean(np.array(abs_pcks))
        pck = np.mean(np.array(pcks))
        vertex_error = np.mean(np.array(vertex_errors))
        return abs_error, error, error_pa, abs_pck, pck, imnames, joints, joints_2ds, vertex_error

def EvalPoseSeg(model, evaltool, loader, viz=False, device=torch.device('cpu')):
    print('-' * 10 + 'evaluation' + '-' * 10)
    seg_results, alpha_mpjpes, pred_mpjpes= [], [], []
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            batchsize = data['img'].size(0)
            if torch.cuda.is_available():
                rgb_img = Variable(data['img']).to(device)
                full_hm = [Variable(item).to(device) for item in data['fullheat']]
            else:
                print('CUDA error')
                sys.exit(0)
            
            # forward
            output = model.model(rgb_img, full_hm)
            seg_result, alpha_mpjpe, pred_mpjpe = evaltool.eval_poseseg(output, data)
            seg_results += seg_result
            alpha_mpjpes += alpha_mpjpe
            pred_mpjpes += pred_mpjpe

            # if i > 1:
            #     break
            # save results
            if i < 4:
                results = {}
                results.update(premask=output['premask'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(preheat=output['preheat'][-1].detach().cpu().numpy().astype(np.float32))
                results.update(heatmap=output['heatmap'].detach().cpu().numpy().astype(np.float32))
                results.update(rgb_img=rgb_img.detach().cpu().numpy().astype(np.float32))
                model.save_results(results, i, batchsize)
        
        alpha_mpjpe = np.mean(np.array(alpha_mpjpes))
        pred_mpjpe = np.mean(np.array(pred_mpjpes))

        return seg_results, alpha_mpjpe, pred_mpjpe


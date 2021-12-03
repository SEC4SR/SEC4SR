
from torch.utils.data import DataLoader
import torch
import pickle
import numpy as np

from metric.metric import get_all_metric

from model.ivector_PLDA import ivector_PLDA
from model.xvector_PLDA import xvector_PLDA
from model.AudioNet import AudioNet

from dataset.Dataset import Dataset

from defense.defense import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bits = 16

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-system_type', type=str, required=True, choices=['audionet', 'iv', 'xv'])
    parser.add_argument('-model_file', type=str, required=True) # spk_model for iv/xv, and ckpt for audionet
    parser.add_argument('-threshold', type=float, default=None) # for SV/OSI task
    
    # for iv-plda
    parser.add_argument('-plda_iv', default='./iv_system/plda.txt')
    parser.add_argument('-mean_iv', default='./iv_system/mean.vec')
    parser.add_argument('-transform_iv', default='./iv_system/transform.txt')
    parser.add_argument('-extractor_iv', default='./iv_system/final_ie.txt')
    parser.add_argument('-gmm', default='./iv_system/final_ubm.txt')

    # for xv-plda
    parser.add_argument('-plda_xv', default='./xv_system/plda.txt')
    parser.add_argument('-mean_xv', default='./xv_system/mean.vec')
    parser.add_argument('-transform_xv', default='./xv_system/transform.txt')
    parser.add_argument('-extractor_xv', default='./xv_system/xvecTDNN_origin.ckpt')

    # for audionet
    parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    #### add a defense layer in the model
    #### Note that for white-box attack, the defense method needs to be differentiable
    parser.add_argument('-defense', default=None, choices=Input_Transformation)
    parser.add_argument('-defense_param', default=None, nargs="+") ### defense method param

    parser.add_argument('-root', type=str, required=True)
    parser.add_argument('-name', type=str, required=True)
    parser.add_argument('-root_ori', type=str, default=None) # directory where the name_ori locates
    parser.add_argument('-name_ori', type=str, default=None) # used to calculate imperceptibility
    parser.add_argument('-wav_length', type=int, default=None)

    ## common attack parameters
    # parser.add_argument('-targeted', action='store_true', default=False)
    parser.add_argument('-batch_size', type=int, default=1)

    parser.add_argument('-target_label_file', default=None) # used to test the targeted attack success rate

    args = parser.parse_args()
    return args

def main(args):

    # load pretrained model
    defense_param = parser_defense_param(args.defense, args.defense_param)
    model = None
    spk_ids = None
    dataset = None
    if args.system_type == 'audionet':
        model = AudioNet(args.label_encoder,
                        transform_layer=args.defense, 
                        transform_param=defense_param)
        # state_dict = torch.load(args.model_file, map_location=device).state_dict()
        state_dict = torch.load(args.model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
    elif args.system_type == 'iv':
        model = ivector_PLDA(args.model_file, args.gmm, args.extractor_iv, 
                args.plda_iv, args.mean_iv, args.transform_iv, device=device, threshold=args.threshold,
                transform_layer=args.defense, transform_param=defense_param)
    elif args.system_type == 'xv':
        model = xvector_PLDA(args.model_file, args.extractor_xv, args.plda_xv, args.mean_xv, args.transform_xv, 
                device=device, threshold=args.threshold,
                transform_layer=args.defense, transform_param=defense_param)
    else:
        raise NotImplementedError('Not Supported Model Type')
    spk_ids = model.spk_ids
    
    wav_length = None if args.batch_size == 1 else args.wav_length
    # If you want to test the distance between ori and adv voices, you must make sure
    # the ori voice is not padded or cutted during adv voice generation, i.e., 
    # no batch attack (batch_size=1) and wav_length=None in attackMain.py
    # The reason is that if ori voice is not padded or cutted, the ori and adv voices will no longer align with each other,
    # and the impercpetibility result will be wrong
    if args.root_ori is not None and args.name_ori is not None:
        wav_length = None
        args.batch_size = 1 # force set args.batch_size to 1
        Warning('You want to test the imperceptibility. \
        Make sure you set batch_size to 1 and wav_length to None for attackMain.py when generating adv. voices \
            Otherwise, the adv. and ori. voices will not align with each other. \
                and the imperceptibility result is wrong.')

    dataset = Dataset(spk_ids, args.root, args.name, return_file_name=True, wav_length=wav_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    if args.root_ori is not None and args.name_ori is not None:
        ori_dataset = Dataset(spk_ids, args.root_ori, args.name_ori, return_file_name=True, wav_length=wav_length)
        ori_loader = DataLoader(ori_dataset, batch_size=args.batch_size, num_workers=0)
        name2ori = {}
        for index, (origin, _, file_name) in enumerate(ori_loader):
            origin = origin.to(device)
            name2ori[file_name[0]] = origin # single audio, since args.batch_size=1

    if args.target_label_file is not None:
        with open(args.target_label_file, 'rb') as reader:
            name2target = pickle.load(reader)

    right_cnt = 0
    target_success_cnt = 0
    imper = [] # imperceptibilty results
    with torch.no_grad():
        for index, (adver, true, file_name) in enumerate(loader):
            adver = adver.to(device)
            true = true.to(device)
            decisions, _ = model.make_decision(adver)
            right_cnt += torch.where(true == decisions)[0].shape[0]
            # print('*' * 10, index, '*' * 10)
            # print(true, decisions) 

            # get target label
            target = None
            if args.target_label_file is not None:
                target = true.clone()
                for ii, name in enumerate(file_name):
                    if name in name2target.keys():
                        target[ii] = name2target[name]
                    else:
                        raise NotImplementedError('Wrong target label file')
                # print(target)
                target_success_cnt += torch.where(target == decisions)[0].shape[0]

            # get original audios
            if args.root_ori is not None and args.name_ori is not None:
                imper_ = get_all_metric(name2ori[file_name[0]], adver)
                imper.append(imper_)
                # print(imper_)
            
            print((f"index: {index} true: {true} target: {target} decision: {decisions}"), end='\r')
    
    print()
    total_cnt = len(dataset)
    ACC = right_cnt * 100 / total_cnt
    print('Acc:', ACC)
    untar_ASR = 100. - ACC
    print('Untargeted Attack Success Rate:', untar_ASR)
    if args.target_label_file is not None:
        target_ASR = target_success_cnt * 100 / total_cnt
        print('Targeted Attack Success Rate:', target_ASR)
    if args.root_ori is not None and args.name_ori is not None:
        imper = np.mean(np.array(imper), axis=0)
        print('L2, L0, L1, SNR, PESQ, STOI', imper)
    

if __name__ == "__main__":

    main(parse_args())
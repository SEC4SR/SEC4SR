
import torch
import numpy as np
import pickle
from torch.utils.data.dataloader import DataLoader

from defense.defense import *

from model.xvector_PLDA import xvector_PLDA
from model.ivector_PLDA import ivector_PLDA
from model.AudioNet import AudioNet

from dataset.Dataset import Dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    
    #Step1: load the model
    defense_param = parser_defense_param(args.defense, args.defense_param)
    if args.task != 'SV' and args.task != 'OSI':
        args.threshold = None # threshold is meaningless for CSI and CSI-NE
    if args.system_type == 'iv':
        model = ivector_PLDA(args.spk_model, args.gmm, args.extractor, 
                args.plda, args.mean, args.transform, device=device, threshold=args.threshold,
                transform_layer=args.defense, transform_param=defense_param)
        spk_ids = model.spk_ids
    elif args.system_type == 'xv':
        model = xvector_PLDA(args.spk_model, args.extractor, args.plda, args.mean, args.transform, 
                device=device, threshold=args.threshold,
                transform_layer=args.defense, transform_param=defense_param)
        spk_ids = model.spk_ids
    elif args.system_type == 'audionet':
        model = AudioNet(args.label_encoder,
                        transform_layer=args.defense, 
                        transform_param=defense_param)
        # state_dict = torch.load(args.model_file, map_location=device).state_dict()
        state_dict = torch.load(args.model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
    else:
        raise NotImplementedError('Unsupported System Type')
    spk_ids = model.spk_ids

    possible_decisions = list(range(len(spk_ids)))
    if args.task == 'SV' or args.task == 'OSI':
        possible_decisions.append(-1) # -1: rejecting

    #Step2: load the dataset
    dataset = Dataset(spk_ids, args.root, args.name, return_file_name=True)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    if args.task == 'SV':
        args.hardest = False # hardest is meaningless for SV task
    if args.hardest and args.simplest:
        args.hardest = False
        args.simplest = False
        Warning('You set both hardest and simplest to true, will roll back to random!!')
        
    #Step3: start
    name2target = {}
    with torch.no_grad():
        for index, (origin, true, file_name) in enumerate(loader):
            origin = origin.to(device)
            true = true.cpu().item()
            decision, scores = model.make_decision(origin)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            candidate_target_labels = [ii for ii in possible_decisions if ii != true and ii != decision]
            candidate_scores = [score for ii, score in enumerate(scores) if ii != true and ii != decision]
            if len(candidate_target_labels) == 0:
                candidate_target_labels = [ii for ii in possible_decisions if ii != decision]
            if len(candidate_scores) == 0:
                candidate_scores = [score for ii, score in enumerate(scores) if ii != decision]
            if not args.hardest and not args.simplest:
                target_label = np.random.choice(candidate_target_labels)
            else:
                if -1 in candidate_target_labels:
                    candidate_target_labels.remove(-1) # reject decision has no score, so remove it
                target_label = candidate_target_labels[np.argmin(candidate_scores)] if args.hardest else \
                    candidate_target_labels[np.argmax(candidate_scores)]
            name2target[file_name[0]] = target_label
            print(index, file_name[0], scores, true, decision, target_label) 
    # Step4: save
    save_path = args.save_path if args.save_path else \
        '{}-{}-{}-{}-{}-{}.target_label'.format(args.system_type, args.task, 
        args.defense, args.defense_param, args.name, args.hardest)
    with open(save_path, 'wb') as writer:
        pickle.dump(name2target, writer, -1)
    print('save file name and target label pair in {}'.format(save_path))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', required=True) # the directory where the dataset locates
    parser.add_argument('-name', required=True) # the dataset name we specify target label for
    
    parser.add_argument('-defense', default=None, choices=(Input_Transformation + [None]))
    parser.add_argument('-defense_param', default=None, nargs="+") ### defense method param

    parser.add_argument('-save_path', default=None) # the path to store the file name and target label pair

    # whether setting the target label such that the attack is the hardest (simplest)
    # When both set to False or both true, will setting the target label randomly
    parser.add_argument('-hardest', action='store_true', default=False)
    parser.add_argument('-simplest', action='store_true', default=False)

    subparser = parser.add_subparsers(dest='system_type') # either iv (ivector-PLDA) or xv (xvector-PLDA)

    iv_parser = subparser.add_parser("iv")
    iv_parser.add_argument('-model_file', required=True) # speaker_model_iv or speaker_model_iv_{ID}
    iv_parser.add_argument('-task', default='CSI', choices=['CSI', 'SV', 'OSI'])
    iv_parser.add_argument('-threshold', default=None, type=float)
    iv_parser.add_argument('-plda', default='./iv_system/plda.txt')
    iv_parser.add_argument('-mean', default='./iv_system/mean.vec')
    iv_parser.add_argument('-transform', default='./iv_system/transform.txt')
    iv_parser.add_argument('-extractor', default='./iv_system/final_ie.txt')
    iv_parser.add_argument('-gmm', default='./iv_system/final_ubm.txt')

    xv_parser = subparser.add_parser("xv")
    xv_parser.add_argument('-model_file', required=True) # speaker_model_xv or speaker_model_xv_{ID}
    xv_parser.add_argument('-task', default='CSI', choices=['CSI', 'SV', 'OSI'])
    xv_parser.add_argument('-threshold', default=None, type=float)
    xv_parser.add_argument('-plda', default='./xv_system/plda.txt')
    xv_parser.add_argument('-mean', default='./xv_system/mean.vec')
    xv_parser.add_argument('-transform', default='./xv_system/transform.txt')
    xv_parser.add_argument('-extractor', default='./xv_system/xvecTDNN_origin.ckpt')

    audionet_parser = subparser.add_parser("audionet")
    audionet_parser.add_argument('-model_file', required=True) # ckpt of pre-trained model 
    audionet_parser.add_argument('-task', default='CSI', choices=['CSI']) # CSI means CSI-NE here!
    audionet_parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    args = parser.parse_args()
    main(args)

'''
Part of the code is drawn from https://github.com/FAKEBOB-adversarial-attack/FAKEBOB
Paper: Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems (IEEE S&P 2021)
'''

import torch
from torch.utils.data import DataLoader
import numpy as np

from model.xvector_PLDA import xvector_PLDA
from model.ivector_PLDA import ivector_PLDA

from dataset.Spk10_test import Spk10_test
from dataset.Spk10_imposter import Spk10_imposter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_threshold(score_target, score_untarget):

    if not isinstance(score_target, np.ndarray):
        score_target = np.array(score_target)
    if not isinstance(score_untarget, np.ndarray):
        score_untarget = np.array(score_untarget)

    n_target = score_target.size
    n_untarget = score_untarget.size

    final_threshold = 0.
    min_difference = np.infty
    final_far = 0.
    final_frr = 0.
    for candidate_threshold in score_target:

        frr = np.argwhere(score_target < candidate_threshold).flatten().size * 100 / n_target
        far = np.argwhere(score_untarget >= candidate_threshold).flatten().size * 100 / n_untarget
        difference = np.abs(frr - far)
        if difference < min_difference:
            final_threshold = candidate_threshold
            final_far = far
            final_frr = frr
            min_difference = difference

    return final_threshold, final_frr, final_far

def main(args):

    #Step 1: set up system helper
    if args.system_type == 'iv':
        model = ivector_PLDA(args.model_file, args.gmm, args.extractor, 
                args.plda, args.mean, args.transform, device=device, threshold=None)
    elif args.system_type == 'xv':
        model = xvector_PLDA(args.model_file, args.extractor, args.plda, args.mean, args.transform, 
                device=device, threshold=None)
    else:
        raise NotImplementedError('Unsupported System Type')
    
    #Step2: load dataset
    test_dataset = Spk10_test(model.spk_ids, args.root, return_file_name=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    imposter_dataset = Spk10_imposter(model.spk_ids, args.root, return_file_name=True)
    imposter_loader = DataLoader(imposter_dataset, batch_size=1, num_workers=0)

    #Step3: scoring
    score_target = []
    score_untarget = []
    trues = [] # used to calculate IER for OSI
    max_scores = [] # used to calculate IER for OSI
    decisions = [] # used to calculate IER for OSI
    with torch.no_grad():
        for index, (origin, true, file_name) in enumerate(test_loader):
            origin = origin.to(device)
            true = true.cpu().item()
            decision, scores = model.make_decision(origin)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            print(index, file_name[0], scores, true, decision)
            if args.task == 'SV':
                score_target.append(scores[true])
                score_untarget += np.delete(scores, true).tolist()
            elif args.task == 'OSI':
                if decision == true:
                    score_target.append(scores[true])
                trues.append(true)
                max_scores.append(np.max(scores))
                decisions.append(decision)

        for index, (origin, true, file_name) in enumerate(imposter_loader):
            origin = origin.to(device)
            true = true.cpu().item()
            decision, scores = model.make_decision(origin)
            decision = decision.cpu().item()
            scores = scores.cpu().numpy().flatten() # (n_spks,)
            print(index, file_name[0], scores, true, decision)
            if args.task == 'SV':
                score_untarget += scores.tolist()
            elif args.task == 'OSI':
                score_untarget.append(np.max(scores))
    
    threshold, frr, far = set_threshold(score_target, score_untarget)
    if args.task == 'SV':
        print("----- Test of {}-PLDA based {}, result ---> threshold: {} FRR: {}, FAR: {}".format(args.system_type, 
        args.task, threshold, frr, far))
    elif args.task == 'OSI':
        IER_cnt = np.intersect1d(np.argwhere(max_scores >= threshold).flatten(),
                    np.argwhere(decisions != trues).flatten()).flatten().size
        # # IER: Identification Error, 
        # for detail, refer to 'Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems'
        IER = IER_cnt * 100 / len(trues) 
        print("----- Test of {}-PLDA based {}, result ---> threshold: {}, FRR: {}, IER: {}, FAR: {} -----".format(
            args.system_type, args.task, threshold, frr, IER, far))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', default='./data')
    parser.add_argument('-task', required=True, choices=['SV', 'OSI']) #the threshold setting of SV is different from OSI

    subparser = parser.add_subparsers(dest='system_type') # either iv (ivector-PLDA) or xv (xvector-PLDA)

    iv_parser = subparser.add_parser("iv")
    #to set threshold, should be the multiple speaker model, not single speaker model, 
    # no matter what the task is (SV/OSI)
    iv_parser.add_argument('-model_file', default='./model_file/speaker_model_iv') # speaker_model_iv
    iv_parser.add_argument('-plda', default='./iv_system/plda.txt')
    iv_parser.add_argument('-mean', default='./iv_system/mean.vec')
    iv_parser.add_argument('-transform', default='./iv_system/transform.txt')
    iv_parser.add_argument('-extractor', default='./iv_system/final_ie.txt')
    iv_parser.add_argument('-gmm', default='./iv_system/final_ubm.txt')

    xv_parser = subparser.add_parser("xv")
    #to set threshold, should be the multiple speaker model, not single speaker model, 
    # no matter what the task is (SV/OSI)
    xv_parser.add_argument('-model_file', default='./model_file/speaker_model_xv') # speaker_model_xv
    xv_parser.add_argument('-plda', default='./xv_system/plda.txt')
    xv_parser.add_argument('-mean', default='./xv_system/mean.vec')
    xv_parser.add_argument('-transform', default='./xv_system/transform.txt')
    xv_parser.add_argument('-extractor', default='./xv_system/xvecTDNN_origin.ckpt')

    args = parser.parse_args()
    main(args)
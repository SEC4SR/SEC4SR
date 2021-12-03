
import torch
import os
import numpy as np
import torchaudio

from model.xvector_PLDA_helper import xvector_PLDA_helper
from model.ivector_PLDA_helper import ivector_PLDA_helper

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):

    #Step 1: set up system helper
    if args.system_type == 'iv':
        helper = ivector_PLDA_helper(args.gmm, args.extractor, 
                args.plda, args.mean, args.transform, device=device)
    elif args.system_type == 'xv':
        helper = xvector_PLDA_helper(args.extractor, args.plda, args.mean, args.transform, device=device)
    else:
        raise NotImplementedError('Unsupported System Type')
    
    #Step2: scoring
    des_dir = args.model_dir
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    model_info = []

    root = args.root
    enroll_dir = os.path.join(root, 'Spk10_enroll')
    spk_iter = os.listdir(enroll_dir)
    for spk_id in spk_iter:

        spk_dir = os.path.join(enroll_dir, spk_id)
        audio_iter = os.listdir(spk_dir)
        ivector = torch.zeros((helper.embedding_dim, ), device=device)
        num_enroll_utt = 0
        for audio_name in audio_iter:
            audio_path = os.path.join(spk_dir, audio_name)
            audio, _ = torchaudio.load(audio_path)
            audio = audio.to(device) * (2 ** (16-1))
            iv = helper.Extract(audio)
            ivector.data += iv
            num_enroll_utt += 1

        ivector.data = ivector / num_enroll_utt
        ivector = helper.process_ivector(ivector, num_utt=1, simple_length_norm=False, normalize_length=True)
        ivector_path = '{}/{}.{}'.format(des_dir, spk_id, args.system_type)
        torch.save(ivector, ivector_path)

        spk_nontarget_scores = []
        test_dir = os.path.join(root, 'Spk10_test')
        test_spk_iter = os.listdir(test_dir)
        for test_spk_id in test_spk_iter:

            if test_spk_id == spk_id:
                continue
            
            test_spk_dir = os.path.join(test_dir, test_spk_id)
            test_audio_iter = os.listdir(test_spk_dir)
            for name in test_audio_iter:
                test_audio_path = os.path.join(test_spk_dir, name)
                test_audio, _ = torchaudio.load(test_audio_path)
                test_audio = (test_audio.to(device) * (2 ** (16-1))).unsqueeze(0)
                scores = helper.score(test_audio, ivector.unsqueeze(0)).flatten().detach().cpu().item()
                spk_nontarget_scores.append(scores)
                print(spk_id, name, scores)

        z_norm_mean = np.mean(spk_nontarget_scores)
        z_norm_std = np.std(spk_nontarget_scores) 
        
        spk_model_info = "{} {} {} {}".format(spk_id, os.path.abspath(ivector_path), z_norm_mean, z_norm_std)
        model_info.append(spk_model_info)
        spk_model_file = os.path.join(des_dir, 'speaker_model_{}_{}'.format(args.system_type, spk_id))
        np.savetxt(spk_model_file, [spk_model_info], fmt='%s')

    model_file = os.path.join(des_dir, 'speaker_model_{}'.format(args.system_type))
    np.savetxt(model_file, model_info, fmt='%s')


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-model_dir', default='./model_file') # path to store speaker model file
    parser.add_argument('-root', default='./data') # path where the Spk10_enroll and Spk10_test locate

    subparser = parser.add_subparsers(dest='system_type') # either iv (ivector-PLDA) or xv (xvector-PLDA)

    iv_parser = subparser.add_parser("iv")
    iv_parser.add_argument('-plda', default='./iv_system/plda.txt')
    iv_parser.add_argument('-mean', default='./iv_system/mean.vec')
    iv_parser.add_argument('-transform', default='./iv_system/transform.txt')
    iv_parser.add_argument('-extractor', default='./iv_system/final_ie.txt')
    iv_parser.add_argument('-gmm', default='./iv_system/final_ubm.txt') 

    xv_parser = subparser.add_parser("xv")
    xv_parser.add_argument('-plda', default='./xv_system/plda.txt')
    xv_parser.add_argument('-mean', default='./xv_system/mean.vec')
    xv_parser.add_argument('-transform', default='./xv_system/transform.txt')
    xv_parser.add_argument('-extractor', default='./xv_system/xvecTDNN_origin.ckpt')


    args = parser.parse_args()
    main(args)

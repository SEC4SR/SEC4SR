
import numpy as np

from model.AudioNetOri import AudioNetOri


class AudioNet(AudioNetOri):
    """Adaption of AudioNet (arXiv:1807.03418)."""
    def __init__(self, label_encoder, transform_layer=None, transform_param=None):

        # parser label info
        id_label = np.loadtxt(label_encoder, dtype=str, converters={0: lambda s: s[1:-1]})
        id2label = {}
        label2id = {}
        for row in id_label:
            id2label[row[0]] = int(row[1])
            label2id[int(row[1])] = row[0]
        self.spk_ids = [label2id[i] for i in range(len(list(label2id.keys())))]
        self.id2label = id2label
        self.label2id = label2id

        super().__init__(len(self.spk_ids), transform_layer=transform_layer, transform_param=transform_param)
        
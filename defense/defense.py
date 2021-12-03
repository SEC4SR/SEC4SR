
Input_Transformation = [

    'QT', 'AT', 'AS', 'MS', # Time Domain
    'DS', 'LPF', 'BPF', # Frequency Domain
    'OPUS', 'SPEEX', 'AMR', 'AAC_V', 'AAC_C', 'MP3_V', 'MP3_C', # Speech Compression
    'FEATURE_COMPRESSION', # Feature-Level,; Ours
    "FeCo", # Feature-Level,; Ours; abbr
] 

Robust_Training = [
    'AdvT' # adversarial training
]

def parser_defense_param(defense, defense_param):

    if defense == 'FeCo' or defense == 'FEATURE_COMPRESSION': # cl_m, point, ratio, other_param (L2, cos, ts, random) 
        defense_param = [defense_param[0], defense_param[1], float(defense_param[2]), defense_param[3]] 
    elif defense == 'BPF':
        defense_param = [float(defense_param[0]), float(defense_param[1])]
    elif defense == 'DS':
        defense_param = float(defense_param[0])
    elif defense:
        defense_param = int(defense_param[0])
    return defense_param
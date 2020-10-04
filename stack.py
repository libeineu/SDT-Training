import collections
import sys

import torch

def main():
    ckpt = torch.load(sys.argv[1])
    #print(ckpt)
    lst = []
    current_layers = ckpt['args'].encoder_layers
    counter_layer = int(sys.argv[3])
    count_layer = 0
    for k, v in ckpt['model'].items():
        k_split = k.split('.')
        if k_split[0] == 'encoder' and k_split[1] == 'layers' and int(k_split[2]) == current_layers - counter_layer:
            l_id = int(k_split[2])
            k_split[2] = str(l_id + int(sys.argv[3]))
            new_k = '.'.join(k_split)
            lst.append([new_k, v.clone()])
            count_layer += 1
            if count_layer == 13:
                counter_layer -= 1
                count_layer = 0
        if k_split[0] == 'encoder' and k_split[1] == 'history' and k_split[2] == 'layer_norms':
            if int(k_split[3]) == len(ckpt['args'].k)-2:
                l_id = int(k_split[3])
                k_split[3] = str(l_id + 1)
                new_k = '.'.join(k_split)
                lst.append([new_k, v.clone()])
    #exit()
    for k, v in lst:
        ckpt['model'][k] = v

    ckpt['args'].encoder_layers += int(sys.argv[3])

    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    '''
    arg1:the input ckpt
    arg2:the output ckpt
    arg3:the num of stack layers(top)
    '''
    main()
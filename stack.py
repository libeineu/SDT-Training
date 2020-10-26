import collections
import sys

import torch
#Number of parameters for each encoder layer
# rpr-network = 13,base-network = 12
num_of_layerpara = 13
strategy = 1

def main():
    ckpt = torch.load(sys.argv[1])
    lst = []
    # Number of copy encoder layers
    counter_layer = int(sys.argv[3])
    #Copy all layers before,such as 6->12->24->48
    if strategy == 0:
        for k, v in ckpt['model'].items():
            k_split = k.split('.')
            if k_split[0] == 'encoder' and k_split[1] == 'layers':
                l_id = int(k_split[2])
                k_split[2] = str(l_id + ckpt['args'].encoder_layers)
                new_k = '.'.join(k_split)
                lst.append([new_k, v.clone()])
            if k_split[0] == 'encoder' and k_split[1] == 'history' and k_split[2] == 'layer_norms':
                l_id = int(k_split[3])
                k_split[3] = str(l_id + ckpt['args'].encoder_layers)
                new_k = '.'.join(k_split)
                lst.append([new_k, v.clone()])
    #sdt g top-most
    elif strategy == 1:
        current_layers = ckpt['args'].encoder_layers
        count_layer = 0
        for k, v in ckpt['model'].items():
            k_split = k.split('.')
            if k_split[0] == 'encoder' and k_split[1] == 'layers' and int(k_split[2]) == current_layers - counter_layer:
                l_id = int(k_split[2])
                k_split[2] = str(l_id + int(sys.argv[3]))
                new_k = '.'.join(k_split)
                lst.append([new_k, v.clone()])
                count_layer += 1
                if count_layer == num_of_layerpara:
                    counter_layer -= 1
                    count_layer = 0
            if k_split[0] == 'encoder' and k_split[1] == 'history' and k_split[2] == 'layer_norms':
                if int(k_split[3]) == len(ckpt['args'].k)-2:
                    l_id = int(k_split[3])
                    k_split[3] = str(l_id + 1)
                    new_k = '.'.join(k_split)
                    lst.append([new_k, v.clone()])
    # top only
    elif strategy == 2:
        current_layers = ckpt['args'].encoder_layers
        count_layer = 0
        num = 1
        for k, v in ckpt['model'].items():
            k_split = k.split('.')
            if k_split[0] == 'encoder' and k_split[1] == 'layers' and int(k_split[2]) == current_layers - 1:
                l_id = int(k_split[2])
                for i in range(counter_layer):
                    k_split[2] = str(l_id + i + 1)
                    new_k = '.'.join(k_split)
                    lst.append([new_k, v.clone()])
            if k_split[0] == 'encoder' and k_split[1] == 'history' and k_split[2] == 'layer_norms':
                if int(k_split[3]) == len(ckpt['args'].k) - 2:
                    l_id = int(k_split[3])
                    k_split[3] = str(l_id + 1)
                    new_k = '.'.join(k_split)
                    lst.append([new_k, v.clone()])
    #Interpolation no sparse connections
    elif strategy == 3:
        layer = 0
        count = 0
        for k, v in ckpt['model'].items():
            # print(k)
            k_split = k.split('.')
            if k_split[0] == 'encoder' and k_split[1] == 'layers':
                l_id = int(k_split[2]) + layer
                k_split[2] = str(l_id)
                new_k1 = '.'.join(k_split)
                k_split[2] = str(l_id + 1)
                new_k2 = '.'.join(k_split)
                lst.append([new_k1, v])
                lst.append([new_k2, v.clone()])
                count += 1
                if count == 13:
                    layer = layer + 1
                    count = 0
    #exit()
    for k, v in lst:
        ckpt['model'][k] = v


    if strategy == 0 or strategy == 3:
        ckpt['args'].encoder_layers *= 2
    elif strategy == 1 or strategy == 2:
        ckpt['args'].encoder_layers += int(sys.argv[3])
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()

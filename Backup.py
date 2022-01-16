
import torch
import torch.nn as nn

import numpy as np

EPSILON = 1e-10



def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num
    
def com_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = len(frame_list)
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    loss = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    return loss

def com_mag_mse_loss(esti, label, frame_list):
    mask_for_loss = []
    utt_num = esti.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], esti.size()[-1]), dtype=esti.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)
        com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    mag_esti, mag_label = torch.norm(esti, dim=1), torch.norm(label, dim=1)
    loss1 = (((esti - label) * com_mask_for_loss) ** 2).sum() / com_mask_for_loss.sum()
    loss2 = (((mag_esti - mag_label) * mask_for_loss) ** 2).sum() / mask_for_loss.sum()
    return 0.5 * (loss1 + loss2)


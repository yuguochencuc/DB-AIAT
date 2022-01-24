import torch
import time
import os
from Backup_pesq import com_mse_loss, com_mag_mse_loss, pesq_loss
import hdf5storage
import gc
from config_merge import *
tr_batch, tr_epoch,cv_epoch = [], [], []


class Solver(object):

    def __init__(self, data, model, optimizer, args):
        # load args parameters
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.loss_dir = args.loss_dir
        self.model = model
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.best_path = args.best_path
        self.cp_path = args.cp_path
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.print_freq = args.print_freq
        self.is_conti = args.is_conti
        self.conti_path = args.conti_path
        self._reset()

    def _reset(self):
        if self.is_conti:
            checkpoint = torch.load(self.conti_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.prev_cv_loss = checkpoint['cv_loss']
            self.best_cv_loss = checkpoint['best_cv_loss']
            #self.start_epoch = 0
            #self.prev_cv_loss = float("inf")
            #self.best_cv_loss = float("inf")
            self.cv_no_impv = 0
            self.having = False

        else:
        #Reset
            self.start_epoch = 0
            self.prev_cv_loss = float("inf")
            self.best_cv_loss = float("inf")
            self.cv_no_impv = 0
            self.having = False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Begin to train.....")
            self.model.train()
            start = time.time()
            tr_avg_loss = self.run_one_epoch(epoch)
            print('-' * 90)
            print("End of Epoch %d, Time: %4f s, Train_Loss:%5f" % (int(epoch+1), time.time()-start, tr_avg_loss))
            print('-' * 90)


            # Cross cv
            print("Begin Cross Validation....")
            self.model.eval()    # BN and Dropout is off
            cv_avg_loss = self.run_one_epoch(epoch, cross_valid=True)
            print('-' * 90)
            print("Time: %4fs, CV_Loss:%5f" % (time.time() - start, cv_avg_loss))
            print('-' * 90)

            # save checkpoint
            check_point = {}
            check_point['epoch'] = epoch+1
            check_point['model_state_dict'] = self.model.state_dict()
            check_point['optimizer_state_dict'] = self.optimizer.state_dict()
            check_point['tr_loss'] = tr_avg_loss
            check_point['cv_loss'] = cv_avg_loss
            check_point['best_cv_loss'] = self.best_cv_loss
            torch.save(check_point, os.path.join(self.cp_path, 'checkpoint_early_exit_%dth.pth.tar' % (epoch+1)))


            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = cv_avg_loss
            tr_epoch.append(tr_avg_loss)
            cv_epoch.append(cv_avg_loss)

            # save loss
            loss = {}
            loss['tr_loss'] = tr_epoch
            loss['cv_loss'] = cv_epoch
            hdf5storage.savemat(self.loss_dir, loss)

            # Adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 5 and self.early_stop == True:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.cv_no_impv = 0

            if self.having == True:
                optim_state = self.optimizer.state_dict()
                for i in range(len(optim_state['param_groups'])):
                    optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                self.having = False
            self.prev_cv_loss = cv_avg_loss

            if cv_avg_loss < self.best_cv_loss:
                self.best_cv_loss = cv_avg_loss
                torch.save(self.model.state_dict(), self.best_path)
                print("Find better cv model, saving to %s" % os.path.split(self.best_path)[1])

    def run_one_epoch(self, epoch, cross_valid=False):
        def _batch(_, batch_info):
            batch_feat = batch_info.feats.cuda()
            batch_label = batch_info.labels.cuda()
            noisy_phase = torch.atan2(batch_feat[:,-1,:,:], batch_feat[:,0,:,:])
            clean_phase = torch.atan2(batch_label[:,-1,:,:], batch_label[:,0,:,:])
            batch_frame_mask_list = batch_info.frame_mask_list

            # three approachs for feature compression:
            if feat_type is 'normal':
                batch_feat, batch_label = torch.norm(batch_feat, dim=1), torch.norm(batch_label, dim=1)
            elif feat_type is 'sqrt':
                batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.5, (
                    torch.norm(batch_label, dim=1)) ** 0.5
            elif feat_type is 'cubic':
                batch_feat, batch_label = (torch.norm(batch_feat, dim=1)) ** 0.3, (
                    torch.norm(batch_label, dim=1)) ** 0.3
            elif feat_type is 'log_1x':
                batch_feat, batch_label = torch.log(torch.norm(batch_feat, dim=1) + 1), \
                                          torch.log(torch.norm(batch_label, dim=1) + 1)

            batch_feat = torch.stack((batch_feat*torch.cos(noisy_phase), batch_feat*torch.sin(noisy_phase)), dim=1)
            batch_label = torch.stack((batch_label*torch.cos(clean_phase), batch_label*torch.sin(clean_phase)), dim=1)

            esti_list = self.model(batch_feat)

            if not cross_valid:
                batch_loss = com_mag_mse_loss(esti_list, batch_label, batch_frame_mask_list)
                batch_loss_res = batch_loss.item()
                tr_batch.append(batch_loss_res)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            else:
                if is_pesq:
                    batch_loss_res = pesq_loss(esti_list, batch_label, batch_frame_mask_list)
                else:
                    batch_loss = com_mag_mse_loss(esti_list, batch_label, batch_frame_mask_list)
                    batch_loss_res = batch_loss.item()
                tr_batch.append(batch_loss_res)
            return batch_loss_res

        start1 = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_loss_res = _batch(batch_id, batch_info)
            total_loss += batch_loss_res
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch:%d, Iter:%d, the average_loss:%5f, current_loss:%5f, %d ms/batch."
                        % (int(epoch + 1), int(batch_id), total_loss / (batch_id + 1), batch_loss_res,
                            1000 * (time.time() - start1) / (batch_id + 1)))
        return total_loss / (batch_id + 1)

from contextlib import contextmanager
@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)

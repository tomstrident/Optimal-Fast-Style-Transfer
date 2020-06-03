# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.parallel

from torch.utils.data import DataLoader
from network import Vgg16, Vgg19
from datasets import FlyingChairs2Dataset, Hollywood2Dataset, COCODataset, SintelDataset

from network import FastStyleNet

import cv2
import time
import imageio
import numpy as np
from skimage import io, transform

# =============================================================================
class FastStyle():
  def __init__(self, debug=True):
    self.train_dir = 'F:/runs/'
    self.debug = debug
    self.device = 'cuda'
    self.method = []
    
    self.VGG16_MEAN = [0.485, 0.456, 0.406]
    self.VGG16_STD = [0.229, 0.224, 0.225]
    
    self.sid_styles = ['autoportrait', 'edtaonisl', 'composition', 'edtaonisl', 'udnie', 'starry_night']#'candy', 
    
    style_grid = np.arange(0, len(self.sid_styles), dtype=np.float32)
    self.style_id_grid = torch.Tensor(style_grid).to(self.device).float()
    
    if debug and not os.path.exists("debug/"):
      os.mkdir("debug/")
  
  def vectorize_parameters(self, params, n_styles):
    vec_pararms = [p*np.ones(n_styles) for p in np.array(params)]
    return np.array(vec_pararms).T
  
  def concat_id(self, params):
    run_id = ""
    
    for j, p in enumerate(params):
      for pi in p:
        run_id += "_" + self.loss_letters[j] + ("%d" % np.log10(pi))

    return run_id + "/"
  
  def train(self, sid=2, epochs=3, emphasis_parameter=[1e0, 1e1], 
            batchsize=6, learning_rate=1e-3, 
            dset='FC2'):
    
    if isinstance(sid, list):
      styles = [self.sid_styles[sidx] for sidx in sid]
      run_id = "msid%d_ep%d_bs%d_lr%d" % (len(sid), epochs, batchsize, np.log10(learning_rate))
      emphasis_parameter = self.vectorize_parameters(emphasis_parameter, len(sid))
    else:
      styles = [self.sid_styles[sid]]
      run_id = "sid%d_ep%d_bs%d_lr%d" % (sid, epochs, batchsize, np.log10(learning_rate))
      emphasis_parameter = self.vectorize_parameters(emphasis_parameter, 1)
    
    self.train_dir = self.train_dir[:8] + dset + '/' + self.method + '/'
    run_id = self.setup_method(run_id, emphasis_parameter.T)
    adv_train_dir = self.train_dir + run_id
    print(adv_train_dir)
    
    if not os.path.exists(adv_train_dir):
      os.makedirs(adv_train_dir)
     
    if os.path.exists(adv_train_dir + '/epoch_' + str(epochs-1) + '.pth'):
      print('Warning: config already exists! Returning ...')
      return
    
    self.prep_training(batch_sz=batchsize, styles=styles, dset=dset)
    self.adam = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    loss_list = []
    
    n_styles = len(self.styles)
    style_grid = np.arange(0, n_styles)
    style_id_grid = torch.LongTensor(style_grid).to(self.device)
    
    for epoch in range(epochs):
      for itr, (imgs, masks, flows) in enumerate(self.dataloader):
        
        imgs = torch.split(imgs, 3, dim=1)
        
        self.prep_adam(itr)
        
        if n_styles > 1:
          style_id = style_id_grid[np.random.randint(0, n_styles)]
        else:
          style_id = 0
        
        losses, styled_img, loss_string = self.train_method(imgs, masks, flows, emphasis_parameter[style_id], style_id)
        
        self.adam.step()
        
        if (itr+1)%1000 == 0:
          torch.save(self.model.state_dict(), '%sfinal_epoch_%d_itr_%d.pth' % (adv_train_dir, epoch, itr//1000))

        if (itr)%1000 == 0 and self.debug:
          imageio.imsave('debug/%d_%d_img1.png' % (epoch, itr), imgs[0].cpu().numpy()[0].transpose(1,2,0))
          imageio.imsave('debug/%d_%d_styled_img1.png' % (epoch, itr), styled_img.detach().cpu().numpy()[0].transpose(1,2,0))
          
        out_string = "[%d/%d][%d/%d] sid%d" % (epoch, epochs, itr, len(self.dataloader), style_id)
        print(out_string + loss_string)
        loss_list.append(torch.FloatTensor(losses).detach().cpu().numpy())

      torch.save(self.model.state_dict(), '%sepoch_%d.pth' % (adv_train_dir, epoch))

    loss_list = np.array(loss_list)
    np.save(adv_train_dir + "loss_list.npy", loss_list)

  #============================================================================
  def infer(self, sid, n_styles, epochs, n_epochs, emphasis_parameter,
            batchsize=6, learning_rate=1e-3,
            dset='FC2', sintel_id='temple_2', sintel_path='D:/Datasets/', 
            vid_fps=20, out_img_path=None, out_img_num=[10]):
    
    if n_styles > 1:
      run_id = "msid%d_ep%d_bs%d_lr%d" % (n_styles, epochs, batchsize, np.log10(learning_rate))
      
    else:
      run_id = "sid%d_ep%d_bs%d_lr%d" % (sid, epochs, batchsize, np.log10(learning_rate))
      
    emphasis_parameter = self.vectorize_parameters(emphasis_parameter, n_styles)
    
    self.train_dir = self.train_dir[:8] + dset + '/' + self.method + '/'
    run_id = self.setup_method(run_id, emphasis_parameter.T)
    #infer_id = run_id[:4] + str(sid) + run_id[5:-1]
    
    print(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth')
    self.model.load_state_dict(torch.load(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth'))
    
    writer = imageio.get_writer('styled_' + self.method + '.mp4', fps=vid_fps)
    dataloader = DataLoader(SintelDataset(sintel_path, sintel_id), batch_size=1)
      
    warped = []
    mask = []
    
    cst_list = []
    lt_cst_list = []
    
    ft_count = []
    styled_list = []
    
    #debug_path = 'C:/Users/Tom/Documents/Python Scripts/Masters Project/debug/'

    style_grid = np.arange(0, len(self.sid_styles), dtype=np.float32)
    style_id_grid = torch.Tensor(style_grid).to(self.device).float()
    style_id = style_id_grid[sid]

    for itr, (frame, mask, flow, lt_data) in enumerate(dataloader):

      if itr > 0:
        flow = flow[0].permute(1, 2, 0).cpu().numpy()
        warped = self.warp_image(styled_list[-1], flow)
      
      t_start = time.time()
      torch_output = self.infer_method((frame, mask, warped), style_id)
      t_end = time.time()
      
      ft_count.append(t_end - t_start)
      
      torch_output = torch.clamp(torch_output, 0.0, 1.0)
      styled_frame = torch_output[0].permute(1, 2, 0).detach().cpu().numpy()
      
      #imageio.imwrite(debug_path + '/img' + str(itr) + '.png', (styled_frame*255.0).astype(np.uint8))
      
      if itr > 0:
        #imageio.imwrite(debug_path + '/warp' + str(itr) + '.png', (warped*255.0).astype(np.uint8))
        #imageio.imwrite(debug_path + '/mask' + str(itr) + '.png', (mask*255.0).astype(np.uint8))
        mask = mask[0].permute(1, 2, 0).cpu().numpy()
        cst = ((mask*(warped - styled_frame))**2).mean()
        cst_list.append(cst)
        #print('FPS:', 1/ft_count[-1], 'CST:', cst_list[-1])
      
      styled_list.append(styled_frame)
      
      lt_len = 5
      if not (itr - lt_len < 0 or itr == len(dataloader) - 1):
        lt_flow, lt_mask = lt_data
        lt_flow = lt_flow[0].permute(1, 2, 0).cpu().numpy()
        lt_mask = lt_mask[0].permute(1, 2, 0).cpu().numpy()
        f_idx2 = itr-lt_len+1
        #imageio.imwrite(debug_path + '/styled_frame2.png', (styled_list[f_idx1]*255.0).astype(np.uint8))
        #imageio.imwrite(debug_path + '/styled_frame1.png', (styled_list[f_idx2]*255.0).astype(np.uint8))
        warped = self.warp_image(styled_list[f_idx2], lt_flow)
        #imageio.imwrite(debug_path + '/warp' + '.png', (warped*255.0).astype(np.uint8))
        #imageio.imwrite(debug_path + '/wmask' + '.png', (lt_mask*255.0).astype(np.uint8))
        
        lt_cst = ((lt_mask[0]*(warped - styled_frame))**2).mean()
        lt_cst_list.append(lt_cst)
      
      real_fid = len(dataloader) - 1 - itr
      if out_img_path != None and real_fid in out_img_num:
        #imageio.imwrite(self.train_dir + infer_path + '_c.png', (np_f*255.0).astype(np.uint8))
        print(out_img_path + dset + "_" + run_id[:-1] + "_" + str(real_fid) + ".png")
        imageio.imwrite(out_img_path + dset + "_" + run_id[:-1] + "_" + str(real_fid) + ".png", (styled_frame*255.0).astype(np.uint8))
      
      cv2.imshow('frame', styled_frame[:,:,[2, 1, 0]])
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      #writer.append_data((styled_frame*255.0).astype(np.uint8))
    
    cv2.destroyAllWindows()
    
    for styled_frame in styled_list[::-1]:
      writer.append_data((styled_frame*255.0).astype(np.uint8))
    
    writer.close()
    
    ft_count = np.array(ft_count[3:])
    fps_count = np.array([1/x for x in ft_count])
    
    avg_ft = ft_count.mean()
    avg_fps = fps_count.mean()
    
    #avg_ft = ft_count.mean()
    #opl_ft = np.percentile(np.sort(ft_count), 1)
    
    #avg_fps = fps_count.mean()
    #opl_fps = np.percentile(np.sort(fps_count), 1)
    
    #oph_ft = self.high_percentile(ft_count, 5)
    #opl_fps2 = self.high_percentile(fps_count, 5)
    
    mse_cst = (np.array(cst_list).mean())**0.5
    mse_lt_cst = (np.array(lt_cst_list).mean())**0.5
    
    print('consistency mse:', mse_cst)
    print('lt consistency mse:', mse_lt_cst)
    print('avg ft:', avg_ft*1000, avg_fps)
    #print('opl ft:', opl_ft*1000, 1/opl_ft, oph_ft, opl_fps, opl_fps2)
    
    return avg_ft*1000, avg_fps, mse_cst, mse_lt_cst
  
  def setup_train(self):
    raise NotImplementedError("Please Implement this method")
  
  def train_method(self):
    raise NotImplementedError("Please Implement this method")
    
  def infer_method(self):
    raise NotImplementedError("Please Implement this method")

  def setup_method(self):
    raise NotImplementedError("Please Implement this method")

  def loadStyles(self, style_name_list, style_size=512):
    styles = []
    
    for i, style_name in enumerate(style_name_list):
      style = io.imread('styles/' + style_name + '.jpg')
      style = torch.from_numpy(transform.resize(style, (style_size, style_size))).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
    
      if self.debug:
        imageio.imsave('debug/0_0_style_' + str(i) + '.png', style.cpu().numpy()[0].transpose(1,2,0))
    
      style = self.normalize(style)
      styled_featuresR = self.vgg(style)
      style_GM = [self.gram_matrix(f) for f in styled_featuresR]
      
      styles.append(style_GM)
    
    return styles

  def load_model(self, model_path):
    print('loading model ...')
    self.model.load_state_dict(torch.load(self.train_dir + model_path))

  def prep_training(self, batch_sz=6, styles=['composition'], dset='FC2'):
    dset_path = 'F:/Datasets/' + dset + '/'
    
    if dset == 'FC2':
      self.dataloader = DataLoader(FlyingChairs2Dataset(dset_path, batch_sz), batch_size=batch_sz)
    elif dset == 'HW2':
      self.dataloader = DataLoader(Hollywood2Dataset(dset_path, batch_sz), batch_size=batch_sz)
    elif dset == 'CO2':
      self.dataloader = DataLoader(COCODataset(dset_path, batch_sz), batch_size=batch_sz)
    else:
      assert False, "Invalid dataset specified error!"
    
    self.train_dir = self.train_dir[:5] + dset + '/'
  
    self.L2distance = nn.MSELoss().to(self.device)
    self.L2distancematrix = nn.MSELoss(reduction='none').to(self.device)
    self.vgg = Vgg16().to(self.device)
    #self.vgg = Vgg19().to(self.device)
    
    for param in self.vgg.parameters():
      param.requires_grad = False
    
    self.styles = self.loadStyles(styles)
    self.adam = []

  def prep_adam(self, itr, batch_sz=1):
    self.adam.zero_grad()

    if (itr+1) % np.int32(500 / batch_sz) == 0:
      for param in self.adam.param_groups:
        param['lr'] = max(param['lr']/1.2, 1e-4)
  
  def calc_tv_loss(self, I):
    sij = I[:, :, :-1, :-1]
    si1j = I[:, :, :-1, 1:]
    sij1 = I[:, :, 1:, :-1]
    
    tv_mat1 = torch.norm(sij1 - sij, dim=1)**2
    tv_mat2 = torch.norm(si1j - sij, dim=1)**2
    
    return torch.sum((tv_mat1 + tv_mat2)**0.5)
  
  def load_mp4(self, video_path):
    reader = imageio.get_reader(video_path + '.mp4')
    fps = reader.get_meta_data()['fps']
    num_f = reader.count_frames()
    print(num_f)
    
    return num_f, fps, reader

  def gram_matrix(self, inp):
    b, c, h, w = inp.size()
    features = inp.view(b, c, h*w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(h*w)
  
  def normalize(self, img):
    mean = img.new_tensor(self.VGG16_MEAN).view(-1, 1, 1)
    std = img.new_tensor(self.VGG16_STD).view(-1, 1, 1)
    return (img - mean) / std

  def warp_image(self, A, flow):
    h, w = flow.shape[:2]
    x = (flow[...,0] + np.arange(w)).astype(A.dtype)
    y = (flow[...,1] + np.arange(h)[:,np.newaxis]).astype(A.dtype)
  
    W_m = cv2.remap(A, x, y, cv2.INTER_LINEAR)
  
    return W_m.reshape(A.shape)

  def styleFrame(self, frame, sid):
    style_id = torch.from_numpy(np.float32([sid])).to(self.device).float()[0]
    
    torch_f = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).float().unsqueeze(0)
    torch_m = torch.zeros(1, 1, frame.shape[0], frame.shape[1])
    torch_w = torch_f
    torch_output = self.infer_method((torch_f, torch_m, torch_w), style_id)
    
    torch_output = torch.clamp(torch_output, 0.0, 1.0)
    styled_frame = torch_output[0].permute(1, 2, 0).detach().cpu().numpy()
    
    return styled_frame
  
  def loadModel(self, sid, n_styles, epochs, n_epochs, emphasis_parameter, 
                batchsize=6, learning_rate=1e-3, dset='FC2'):
    
    if n_styles > 1:
      run_id = "msid%d_ep%d_bs%d_lr%d" % (n_styles, epochs, batchsize, np.log10(learning_rate))
      
    else:
      run_id = "sid%d_ep%d_bs%d_lr%d" % (sid, epochs, batchsize, np.log10(learning_rate))
      
    emphasis_parameter = self.vectorize_parameters(emphasis_parameter, n_styles)
    
    self.train_dir = self.train_dir[:8] + dset + '/' + self.method + '/'
    run_id = self.setup_method(run_id, emphasis_parameter.T)
    
    print(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth')
    self.loadModelID(self.train_dir + run_id + 'epoch_' + str(n_epochs) + '.pth')
    
  def loadModelID(self, n_styles, model_id):
    self.model = FastStyleNet(3, n_styles).to(self.device)
    self.model.load_state_dict(torch.load(model_id))
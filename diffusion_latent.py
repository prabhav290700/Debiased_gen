from audioop import reverse
import math
from genericpath import isfile
from glob import glob
from models.guided_diffusion.script_util import guided_Diffusion
from models.improved_ddpm.nn import normalization
from tqdm import tqdm
import os
import numpy as np
#import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from torchvision import models
import torchvision.transforms as transforms
import torch.nn.functional as F
from losses.clip_loss import CLIPLoss
import random
import copy
import matplotlib.pyplot as plt

import clip
# from models.ddpm.diffusion import DDPM
# from models.improved_ddpm.script_util import i_DDPM
from utils.diffusion_utils import get_beta_schedule, denoising_step
from utils.text_dic import SRC_TRG_TXT_DIC
from losses import id_loss
# from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS
# from datasets.imagenet_dic import IMAGENET_DIC

counter=0

def conversion_timestep(t):
    array = [0, 20, 40, 61, 81, 101, 122, 142, 163, 183, 203, 224, 244, 265, 285, 305, 326, 346, 366, 387, 407, 428, 448, 468, 489, 509, 530, 550, 570, 591, 611, 632, 652, 672, 693, 713, 733, 754, 774, 795, 815, 835, 856, 876, 897, 917, 937, 958, 978, 978] # array length = 50
    return array[t-1]

idx_to_attr_dict = {
    0: "eyeglasses",
    1: "gender",
    2: "race",
    3: "age",
    4: "race_multi"
}

attr_class_dict = {
    "gender": ['male','female'],
    "eyeglasses": ['0_glasses','1_glasses'],
    "race": ['white','black'],
    "age": ['young','middle_aged','old'],
    "race_multi": ['white','black','asian','brown']
}


class Asyrp(object):
    def __init__(self, args, config):
        # ----------- predefined parameters -----------#
        self.args = args
        self.config = config
        #if device is None:
        #     device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        self.device = config.device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        self.learn_sigma = False 

    def load_pretrained_model(self):

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset in ["CelebA_HQ", "CUSTOM", "CelebA_HQ_Dialog"]:
            # Idk? maybe SDE2
            url = "./pretrained/celebahq_p2.pt"
           
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET", "MetFACE","CelebA_HQ_P2"]:
            pass
        else:
            raise ValueError
        if self.config.data.dataset in ["CelebA_HQ", "LSUN", "CelebA_HQ_Dialog"]:
            model = DDPM(self.config) 
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(url, map_location=self.device)
            self.learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            model = i_DDPM(self.config.data.dataset) #Get_h(self.config, model="i_DDPM", layer_num=self.args.get_h_num) #
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            self.learn_sigma = True
            print("Improved diffusion Model loaded.")
        elif self.config.data.dataset in ["MetFACE", "CelebA_HQ_P2"]:
            model = guided_Diffusion(self.config.data.dataset)
            init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            print("Model loaded", self.config.data.dataset)
            self.learn_sigma = True
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt, strict=False)

        return model

    
    @torch.no_grad()
    def save_image(self, model, clip_ver, texts, x_lat_tensor, seq_inv, seq_inv_next, save_process_origin = False, get_delta_hs=False,
                    folder_dir="", save_result_dir="", hs_coeff=(1.0,1.0)):

        global counter
        
        clipModel = None
        text_features = None

        if self.args.debias in ["image_space", "h_space"]:
            clipModel, _ = clip.load(clip_ver, device=self.config.device)
            
        
        with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
            x_list = []

            labels = None
    
            x = x_lat_tensor.clone().to(self.device)

            for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                t = (torch.ones(1) * i).to(self.device)
                t_next = (torch.ones(1) * j).to(self.device)

            
                first_debiasing_step = it == 0
                
                x, x0_t, grads, _  = denoising_step(x, t=t, 
                                t_next=t_next,
                                editList=self.args.editList,
                                tagtime=self.args.tagtime,
                                n_control=self.args.n_control, 
                                models=model,
                                clipModel=clipModel,
                                text=texts,
                                strn=self.args.strength,
                                p=self.args.proportions,
                                logvars=self.logvar,
                                sample = self.args.sample,
                                male = self.args.male,
                                eyeglasses = self.args.eyeglasses,
                                scale = self.args.scale,
                                timestep_list = self.args.timestep_list,
                                # usefancy = self.args.usefancy,
                                # gamma_factor = self.args.gamma_factor,
                                # guidance_loss = self.args.guidance_loss,
                                attribute_list = self.args.attribute_list,
                                vanilla_generation = self.args.vanilla_generation,
                                debias=self.args.debias,
                                first_debiasing_step = first_debiasing_step,
                                universal_guidance=self.args.universal_guidance,
                                sampling_type= self.args.sample_type,
                                b=self.betas,
                                learn_sigma=self.learn_sigma,
                                eta=1.0
                                )
                progress_bar.update(1)


            
            x_list.append(x)
                


            

        x = torch.cat(x_list, dim=0)


        x = (x + 1) * 0.5

        
        # directory_path = os.path.join("Results",f"{self.args.seed}",f"{self.args.bs_test}", f"{save_result_dir}{self.args.debias}",f"{self.args.n_control}_{self.args.strength}_{self.args.editList}_{self.args.tagtime}")
        # os.makedirs(directory_path, exist_ok=True)
        for idx, image in enumerate(x):
            image_path = os.path.join(self.args.exp, f'{counter + idx}.png')

            tvu.save_image(image, image_path)
            
        counter += len(x)

        
        
        
    @torch.no_grad()
    def run_test(self,texts,clip_ver='ViT-B/32'):
        
        print("Running Test")
        
        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s+1e-6) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])
        print("loading model pretrained")
        model = self.load_pretrained_model()

            

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        device = next(model.module.parameters()).device
        print(f"device_model: {device}")
        print("model moved to device")

        print("Prepare identity latent...")


        if self.args.just_precompute:
            img_lat_pairs_dic = self.precompute_pairs(model, self.args.save_precomputed_images)
            print("Pre-computed done.")
            return
        else:
            print("using random noise")
            img_lat_pairs_dic = self.random_noise_pairs(model)


        print("number of noise vectors", len(img_lat_pairs_dic["test"]))
        if self.args.target_image_id:
            self.args.target_image_id = self.args.target_image_id.split(" ")
            self.args.target_image_id = [int(i) for i in self.args.target_image_id]

        x_lat_tensor = None
        model.eval()
        
        
        print("inside testing set")
        
        folder_pth=""
        
        attributes = [idx_to_attr_dict[idx] for idx, value in enumerate(self.args.attribute_list) if value == 1]
        for attr in attributes:
            folder_pth=folder_pth+attr+"_" 
#############################################################################        
        
        
        x_lat_tensor = None
        for step, (_, _, x_lat) in enumerate(img_lat_pairs_dic['test']):
            print(f"batch num: {step}")
            if self.args.target_image_id:
                if not step in self.args.target_image_id:
                    continue

            if self.args.start_image_id > step:
                continue

            if x_lat_tensor is None:
                x_lat_tensor = x_lat.to(self.device)
                # x0_tensor = x0
            else:
                x_lat_tensor = torch.cat((x_lat_tensor, x_lat), dim=0)
                
            
            self.save_image(model,clip_ver,texts, x_lat_tensor, seq_test, seq_test_next,
                                        folder_dir=self.args.test_image_folder,
                                        save_process_origin=self.args.save_process_origin,
                                        save_result_dir=folder_pth
                                        )
            
            
                                    
            if (step+1)*self.args.bs_test >= self.args.n_test_img:
                break
            x_lat_tensor = None



    # ----------- Pre-compute -----------#
    @torch.no_grad()
    def precompute_pairs(self, model):
    
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        for mode in ['test']:
            img_lat_pairs = []

        
            print("no path or recompute")

            # if self.config.data.category == 'CUSTOM':
                
                # print("custom:", self.args.custom_train_dataset_dir)
                # DATASET_PATHS["custom_train"] = self.args.custom_train_dataset_dir


            DATASET_PATHS["custom_test"] = os.path.join(self.args.test_path_one)
            print("path to generate:", DATASET_PATHS["custom_test"])

            test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
            loader_dic = get_dataloader(test_dataset, num_workers=self.config.data.num_workers, shuffle=False)
            loader = loader_dic[mode]
            
            # if self.args.save_process_origin:
            #     save_process_folder = os.path.join(self.args.image_folder, f'inversion_process')
            #     if not os.path.exists(save_process_folder):
            #         os.makedirs(save_process_folder)


            for step, (img, label) in enumerate(loader):
                if os.path.exists(os.path.join(self.args.savepath, f'{label[0].split(".")[0]}.pt')):
                    continue
                
                if (mode == "test" and step == self.args.n_test_img):
                    break
                x0 = img.to(self.config.device)
                # if save_imgs:
                #     tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                
                with torch.no_grad():
                    h_vector = []

                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, _, _, h = denoising_step( x, t=t, t_next=t_prev, models=model,
                                            logvars=self.logvar,
                                            sampling_type='ddim',
                                            b=self.betas,
                                            eta=0,
                                            learn_sigma=self.learn_sigma,
                                            )
                            
                            for i in range(len(h)):
                                h_vector.append(h.detach().cpu())
                                
                            progress_bar.update(1)
                    

                    h_vector = torch.cat(h_vector, dim=0)
                    torch.save(h_vector, os.path.join(self.args.savepath, f'{label[0].split(".")[0]}.pt'))

    # ----------- Get random latent -----------#
    @torch.no_grad()
    def random_noise_pairs(self, model):

        print("Prepare random latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s+1e-6) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        img_lat_pairs_dic = {}

        test_lat = []

        for i in range(self.args.n_test_img//self.args.bs_test):
            lat = torch.randn((self.args.bs_test, self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            test_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

        if  self.args.n_test_img%self.args.bs_test!=0:
            lat = torch.randn((self.args.n_test_img%self.args.bs_test, self.config.data.channels, self.config.data.image_size, self.config.data.image_size))
            test_lat.append([torch.zeros_like(lat), torch.zeros_like(lat), lat])

        img_lat_pairs_dic['test'] = test_lat
        
        return img_lat_pairs_dic

    @torch.no_grad()
    def set_t_edit_t_addnoise(self, LPIPS_th=0.33, LPIPS_addnoise_th=0.1, return_clip_loss=False):

        clip_loss_func = CLIPLoss(
            self.device,
            lambda_direction=1,
            lambda_patch=0,
            lambda_global=0,
            lambda_manifold=0,
            lambda_texture=0,
            clip_model=self.args.clip_model_name)

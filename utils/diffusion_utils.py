import numpy as np
import torch
# from Cluster import *
import torchvision.utils as tvu
from torchvision.utils import save_image

import clip

idx_to_attr_dict = {
    0: "eyeglasses",
    1: "gender",
    2: "race",
}

attr_class_dict = {
    "gender": ['male','female'],
    "eyeglasses": ['0_glasses','1_glasses'],
    "race": ['white','black'],
}

def clip_preprocess(image):
    _, clip_preprocess = clip.load('ViT-B/32', device='cuda')

    preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    image_input = preprocess(image)
    return image_input


def conversion_timestep(t):
    array = [0, 20, 40, 61, 81, 101, 122, 142, 163, 183, 203, 224, 244, 265, 285, 305, 326, 346, 366, 387, 407, 428, 448, 468, 489, 509, 530, 550, 570, 591, 611, 632, 652, 672, 693, 713, 733, 754, 774, 795, 815, 835, 856, 876, 897, 917, 937, 958, 978, 978] # array length = 50
    return array[t-1]

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def getTags(p,n):
    target=p*n
    arr = np.repeat(np.arange(len(target)), target)
    #np.random.shuffle(arr)
    return arr

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs = x_shape[0]
    # out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    # print(a, t)
    out = torch.gather(a.clone().detach().to(t.device), 0, t.long())
    # print(out)
    out = torch.full((bs, 1, 1, 1), out.item(), device=t.device)
    return out

tags = None

def denoising_step(xt, t, t_next,
                   editList,
                   tagtime,
                   n_control,
                   *,
                   models,
                   clipModel,
                   text,
                   strn,
                   p,
                   logvars,
                   b,
                   sampling_type='ddim',
                   eta=0.0,
                   learn_sigma=False,
                   sample = True,
                   male = 1,
                   eyeglasses = 1,
                   scale = [1500],
                   timestep_list = [0,50],
                #    usefancy = False,
                #    gamma_factor = 0.1,
                #    guidance_loss = 'chi_without_5',
                   attribute_list = [0,1,0,0],
                   vanilla_generation = False,
                   debias=False,
                   first_debiasing_step = False,
                   t_edit=0,
                   hs_coeff=(1.0),
                   delta_h=None,
                   use_mask=False,
                   dt_lambda=1,
                   ignore_timestep=False,
                   image_space_noise=0,
                   universal_guidance=False,
                   dt_end = 999,
                   warigari=False
                   ):

    global tags
    # Compute noise and variance
    model = models

    # Compute the next x
    bt = extract(b, t, xt.shape)
    at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)

    # (self, x, alpha, clip_model, text_features, timesteps, sample = True, male = 1, eyeglasses = 1, scale = [1500], timestep_list = [1,50], y=None, t_edit=400, hs_coeff=(1.0, 1.0), delta_h=None, ignore_timestep=False , use_mask=False, bt=1, attribute_list=[0,1,0,0], vanilla_generation=False, debias=False)
    # et, et_modified, delta_h, middle_h = model(xt, t, sample=sample, male = male, eyeglasses=eyeglasses, scale=scale, index=index, t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask)
    # et, et_modified, delta_h, middle_h = model(xt, at, clip_model, text_features, t, sample=sample, timestep_list = timestep_list, male = male, eyeglasses=eyeglasses, scale=scale, attribute_list=attribute_list, vanilla_generation=vanilla_generation,debias=debias, bt = bt[0].item(), t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask)
    # et, et_modified, delta_h, middle_h = model(xt, at, clip_model, text_features, t, sample=sample, timestep_list = timestep_list, male = male, eyeglasses=eyeglasses, scale=scale, attribute_list=attribute_list, vanilla_generation=vanilla_generation,debias=debias, bt = bt[0].item(), t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask)
    # p = [0.5, 0.5]

    et, et_modified, delta_h, middle_h = model(xt, at,  text, strn, editList, tagtime, n_control, p, t, clipModel=clipModel, sample=sample, timestep_list = timestep_list, male = male, eyeglasses=eyeglasses, scale=scale, attribute_list=attribute_list, vanilla_generation=vanilla_generation,debias=debias, bt = bt[0].item(), t_edit=t_edit, hs_coeff=hs_coeff, delta_h=delta_h, ignore_timestep=ignore_timestep, use_mask=use_mask) 



    if learn_sigma:
        et, logvar_learned = torch.split(et, et.shape[1] // 2, dim=1)
        # if index is not None:
        #     et_modified, _ = torch.split(et_modified, et_modified.shape[1] // 2, dim=1)
        logvar = logvar_learned
    # else:
    #     logvar = extract(logvars, t, xt.shape)

    # if type(image_space_noise) != int:
    #     print("-----------inside")
    #     if t[0] >= t_edit:
    #         index = 0
    #         if type(image_space_noise) == torch.nn.parameter.Parameter:
    #             et_modified = et + image_space_noise * hs_coeff[1]
    #         else:
    #             # 
    #             # (type(image_space_noise))
    #             temb = models.module.get_temb(t)
    #             et_modified = et + image_space_noise(et, temb) * 0.01

    # # Compute the next x
    # bt = extract(b, t, xt.shape)
    # at = extract((1.0 - b).cumprod(dim=0), t, xt.shape)
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at)
    else:
        at_next = extract((1.0 - b).cumprod(dim=0), t_next, xt.shape)

    xt_next = torch.zeros_like(xt)

    if sampling_type == 'ddpm':
        weight = bt / torch.sqrt(1 - at)

        mean = 1 / torch.sqrt(1.0 - bt) * (xt - weight * et)
        noise = torch.randn_like(xt)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((xt.shape[0],) + (1,) * (len(xt.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()

    elif sampling_type == 'ddim':
        # if index is not None:
        #     x0_t = (xt - et_modified * (1 - at).sqrt()) / at.sqrt()
        # else:
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        # Add noise. When eta is 1 and time step is 1000, it is equal to ddpm.
        else:
            c1 = eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            # noise = torch.randn_like(xt[0:1])
            # print(noise.shape)
            # noise = noise.repeat(xt_next.shape[0],1,1,1)
            # xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * noise
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)

    if dt_lambda != 1 and t[0] >= dt_end:
        xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et * dt_lambda

    # # Asyrp & DiffStyle
    # if not warigari or index is None:
    #     return xt_next, x0_t, delta_h, middle_h

    # # Warigari by young-hyun, Not in the paper
    # else:
    #     # will be updated
    
    # tvu.save_image((x0_t+1)*0.5,f"image_at_{t.item()}.png")
    return xt_next, x0_t, delta_h, middle_h
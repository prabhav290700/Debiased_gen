
# A Multi-Modal framework for Debiasing Diffusion Models using Tweedie’s


### Set Up 
Create the environment by running the following:
```bash
conda env create -f environment.yml
```
Download the pretrained models from [here](https://1drv.ms/u/s!AkQjJhxDm0Fyhqp_4gkYjwVRBe8V_w?e=Et3ITH) and store it in `pretrained/`


Run the `./text.sh` and `./exemplar.sh` files to generate images based on the mode you want to use:
### Execution
<details>
<summary><span style="font-weight: bold;">arguments for generation</span></summary>

- `exp`: Path that the images should be stored in.
- `n_test_img` : # images to be generated
- `attribute_list` : Attribute to be balanced: [1,0,0,0 - Eyeglasses, 0,1,0,0 - Gender, 0,0,1,0 - Race] (For multi attributes, add 1's accordingly, ex: 1,1,0,0 = Eyeglasses+ Gender)
- `bs_test` : batch-size for generation
-  `editList` : sub-window where score guidance is to be applied.
- `tagtime` : time when tags are to be frozen(should lie in editlist)
- `control` : number of updations to be done at each step
- `strength` : Guidance scale Refer to appendix section for  of the attributes in the paper, if not present, needs to be tuned
- `text` : extra argument in `text.sh` for text based generation, description of attribute values.
- `proportions` : extra argument in `text.sh` dictating the distribution over attribute values

to change the proportion for exemplar based generations, you need to change it in `multi_diffusion_latent.py`


</details>


## Acknowledge
Codes are based on [BalancingAct](https://github.com/rishubhpar/debiasing_gen_models)


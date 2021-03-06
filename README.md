# II2S_2


1. Clone this repo https://github.com/rosinality/stylegan2-pytorch.
2. Add these files to *stylegan2-pytorch* 

---
How do random samples map from the Z space to the W and P spaces? 

<p float="left">
  <img src="/Images/ZSpace.png" width="300" />
  <img src="/Images/WSpace.png" width="300" /> 
  <img src="/Images/PSpace.png" width="300" />
</p>

---

<p float="left">
  <img src="/Images/00.png" width="300" />
  <img src="/Images/lambda0001.png" width="300" /> 
</p>

---
Inverting a random image into the different spaces through optimization, using the LPIPS loss!

*Z Space*
<p float="left">
  <img src="/Images/conv.gif" width="240" />
  <img src="/Images/000000.png" width="240" /> 
</p>

*W Sapce*
<p float="left">
  <img src="/Images/WSpace.gif" width="240" />
  <img src="/Images/000000.png" width="240" /> 
</p>

*W+ Space*
<p float="left">
  <img src="/Images/WPlusSpace.gif" width="240" />
  <img src="/Images/000000.png" width="240" /> 
</p>

---

*Pn Space*
Inverting a random image using the LPIPS loss and Pn Loss!

*lambda = 0.01*


<p float="left">
  <img src="/Images/Pn_01.gif" width="240" />
  <img src="/Images/000000.png" width="240" /> 
</p>

*lambda = 0.001*


<p float="left">
  <img src="/Images/Pn_001.gif" width="240" />
  <img src="/Images/000000.png" width="240" /> 
</p>

---
### Idea behind e4e
1. Minimize the variation of the 18 latent codes 
  * predict a single latent code (and offsets for the other 17 codes)
  * make the offsets as small as possible using L2 regularization
2. Encourage each individual style code to be within the W distribution
  * a discriminator learns to distinguish between the real latent vectors sampled from StyleGAN’s mapping network and fakes ones from the encoder

### Idea behind PTI
1. Use direct optimization to invert the image and obtain the pivotal latent code
2. Tune the generator to generate the input image given the latent code in the previous step

* They also use locality regularization to make the tuning effects localized and keep the StyleGAN latent space semantically editable. 
---


# Editability Evaluation:


In our experiments, we noticed that I2S2 with the Pn regularization loss struggles to get editable images when the image resolution is increased to 1024x1024. Other than obvious perceptual loss reason, we believe that changing hyperparameters slightly led to inconsistent results. We tried to investigate different initialization techniques and different hyperparameters. We have noticed that changing the number of components in the PCA transformation gave us better control over convergence, which resulted in more stable results. Below, we summarized the results of our experiments.  


**Images below show Ground Truth, Reconstructed Image, Expression Change, and Age Change (Respectively). 
Lambda refers to the Pn loss coefficient.**


## Images taken from StyleGan domain. 

* Iteration = 1300. 
* lr = 0.01. 
* lambda = 0. 

 
<p float="left">
  
  <img src="/Images/lambda0/000526.png" alt="Nature" width="240" />
  <img src="/Images/pn_0_1_reconstruct.png" width="240" />
  <img src="/Images/lambda0/pn_0_1.png" width="240" />
  <img src="/Images/lambda0/pn_0_11.png" width="240" />
</p>

<p float="left">
  <img src="/Images/lambda0/000509.png" width="240" />
  <img src="/Images/pn_0_2_reconstruct.png" width="240" />

  <img src="/Images/lambda0/pn_0_2.png" width="240" />
  <img src="/Images/lambda0/pn_0_22.png" width="240" />
</p>

<p float="left">
  <img src="/Images/lambda0/000518.png" width="240" />
  <img src="/Images/pn_0_3_reconstruct.png" width="240" />

  <img src="/Images/lambda0/pn_0_3.png" width="240" />
  <img src="/Images/lambda0/pn_0_33.png" width="240" />
</p>

---

* Iteration = 2000. 
* lr = 0.01 and decreases by a factor of 0.8 every 200 iteration.
* Number of Components = 128 (Instead of 512).
* lambda = 0.001 and increases by a factor of 1.15 every 100 iteration.  
* Every other hyperparamter should follow II2S implementation.



<p float="left">
  <img src="/Images/lambda0/000526.png" width="240" />
  
  <img src="/Images/pn128_lr001/pn_128_1_gt.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_1.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_11.png" width="240" />
</p>


<p float="left">
  <img src="/Images/lambda0/000509.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_2_gt.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_2.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_22.png" width="240" />
</p>



<p float="left">
  <img src="/Images/lambda0/000518.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_3_gt.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_33.png" width="240" />
  <img src="/Images/pn128_lr001/pn_128_3.png" width="240" />
</p>

---

* Iteration = 2000. 
* lr = 0.01 and decreases by a factor of 0.8 every 200 iteration.
* Number of Components = 128 (Instead of 512).
* lambda = 0.01 and increases by a factor of 1.15 every 100 iteration.
* Every other hyperparamter should follow II2S implementation.


<p float="left">
  <img src="/Images/lambda0/000526.png" width="240" />
  
  <img src="/Images/pn_128_lr01/pn_1_1.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_11.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_111.png" width="240" />
</p>


<p float="left">
  <img src="/Images/lambda0/000509.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_2_r.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_22.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_2.png" width="240" />
</p>



<p float="left">
  <img src="/Images/lambda0/000518.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_3_r.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_3.png" width="240" />
  <img src="/Images/pn_128_lr01/pn_1_33.png" width="240" />
</p>


---

## Out of Domain Images

1300 iteration. lr = 0.01. 
lambda = 0. Images below show Ground Truth, Reconstructed Image, Expressin Change, and Age Change (Respectively). 

* Iteration = 1300. 
* lr = 0.01.
* lambda = 0.0.  
* Every other hyperparamter should follow II2S implementation.


<p float="left">
  <img src="/Images/ex0/2.png" width="240" />
  <img src="/Images/ex0/ex_recon.png" width="240" />
  <img src="/Images/ex0/ex_exp.png" width="240" />
  <img src="/Images/ex0/ex_age.png" width="240" />
</p>


<p float="left">
  <img src="/Images/ex0/3.png" width="240" />
  <img src="/Images/ex0/ex2_recon.png" width="240" />
  <img src="/Images/ex0/ex2_exp.png" width="240" />
  <img src="/Images/ex0/ex2_age.png" width="240" />
</p>



<p float="left">
  <img src="/Images/ex0/0.png" width="240" />
  <img src="/Images/ex0/ex3_recon.png" width="240" />
  <img src="/Images/ex0/exp3_exp.png" width="240" />
  <img src="/Images/ex0/ex3_age.png" width="240" />
</p>

---

* Iteration = 2000. 
* lr = 0.01 and decreases by a factor of 0.8 every 200 iteration.
* Number of Components = 128 (Instead of 512).
* lambda = 0.001 and increases by a factor of 1.15 every 100 iteration.
* Every other hyperparamter should follow II2S implementation.


<p float="left">
  <img src="/Images/ex0/2.png" width="240" />
  <img src="/Images/ex01/e1_recon.png" width="240" />
  <img src="/Images/ex01/e1_exp.png" width="240" />
  <img src="/Images/ex01/e1_age.png" width="240" />
</p>


<p float="left">
  <img src="/Images/ex0/3.png" width="240" />
  <img src="/Images/ex01/e2_recon.png" width="240" />
  <img src="/Images/ex01/e2_exp.png" width="240" />
  <img src="/Images/ex01/e2_age.png" width="240" />
</p>


<p float="left">
  <img src="/Images/ex0/0.png" width="240" />
  <img src="/Images/ex01/e3_recon.png" width="240" />
  <img src="/Images/ex01/e3_exp.png" width="240" />
  <img src="/Images/ex01/e3_age.png" width="240" />
</p>


---

* Iteration = 2000. 
* lr = 0.01 and decreases by a factor of 0.8 every 200 iteration.
* Number of Components = 128 (Instead of 512).
* lambda = 0.001 and increases by a factor of 1.15 every 100 iteration.
* Every other hyperparamter should follow II2S implementation.


<p float="left">
  <img src="/Images/ex0/2.png" width="240" />
  <img src="/Images/ex64/ee1_recon.png" width="240" />
  <img src="/Images/ex64/ee1_exp.png" width="240" />
  <img src="/Images/ex64/ee1_age.png" width="240" />
</p>



<p float="left">
  <img src="/Images/ex0/3.png" width="240" />
  <img src="/Images/ex64/ee2_recon.png" width="240" />
  <img src="/Images/ex64/ee2_exp.png" width="240" />
  <img src="/Images/ex64/ee2_age.png" width="240" />
</p>


<p float="left">
  <img src="/Images/ex0/0.png" width="240" />
  <img src="/Images/ex64/ee3_recon.png" width="240" />
  <img src="/Images/ex64/ee3_exp.png" width="240" />
  <img src="/Images/ex64/ee3_age.png" width="240" />
</p>



---
# Reconstruction Evaluation:

| Cases      | LPIPS | MSE     | MSSSIM |
| :---:       |    :----:   |         :---: |     :---: |
| lambda = 0.0       | 0.397       | 0.033  | 0.441   |
|lambda = 0.001, #Components = 128| 0.398        | 0.034      | 0.461   |
| lambda = 0.01, #Components = 128| 0.401        | 0.0035      | 0.466 |
| lambda = 0.01, #Components = 64   | 0.398        | 0.033      |  0.460  |
| lambda = 0.1, #Components = 64   | 0.399      | 0.034     | 0.463 |



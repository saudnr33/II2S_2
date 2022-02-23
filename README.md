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





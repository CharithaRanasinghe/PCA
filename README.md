# PCA for AI Generated Image Detection
---
Principle Component Analysis (PCA) can be utilized to distinguish between AI generated images and real images, via reduction of detail. First the input image is converted into an array, then PCA is done in 3 colours seperately for Red, Green and Blue. The result after PCA will have differences with respect to the original image, that difference is taken as the "residual", which is used as the parameter to be analysed through various statistical operations such as mean, deviation, skewness and entropy, finally resulting the amount of predictable existance of AI generated content in the input image.
<img width="1635" height="674" alt="image" src="https://github.com/user-attachments/assets/c78b3330-00b5-462c-8c28-36dd41870e18" />
<img width="1626" height="672" alt="image" src="https://github.com/user-attachments/assets/b94aa7da-5dbf-4611-9941-1026edfbaff5" />

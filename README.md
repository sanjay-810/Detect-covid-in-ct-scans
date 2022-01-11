# Coivd-prediction-using-ct-scans
predict whether covid is present or not using the CT scan. We have taken the dataset from Kaggle. model is unet architecture, giving the output an image that highlights the required features. We have used unet because it was primarily used for biomedical images, where we don't have a vast dataset.
# Details about input data
Input is nii type extension.
![alt text](https://github.com/sanjay-810/Detect-covid-in-ct-scans/blob/main/sample_images/sample%20image%20of%20dataset.png?raw=true)
# Accuracy and loss
In unet architecture, I have used nine layers. Up to four layers. The image is converted into an array with all the possible features; from layer five, the array will be transposed into higher pixel images with pre-layer concatenate with the present layer. After ten epochs, my model accuracy is 0.9961, validation accuracy is 0.9964, and test data accuracy is 0.8721
![alt text](https://github.com/sanjay-810/Detect-covid-in-ct-scans/blob/main/sample_images/loss%20vs%20epochs.png?raw=true)
![alt text](https://github.com/sanjay-810/Detect-covid-in-ct-scans/blob/main/sample_images/epochs%20vs%20acc.png?raw=true)

Predicted Image
![alt text](https://github.com/sanjay-810/Detect-covid-in-ct-scans/blob/main/sample_images/final.png?raw=true)

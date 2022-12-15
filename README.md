# Improving Hemorrhage Detection in Sparse-view CTs via Deep Learning
Code to the paper: "Improving Hemorrhage Detection in Sparse-view CTs via Deep Learning" 

How to use this code:
1) Download the stage_2_train folder from https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data
2) Unzip and place in ./Data/
3) Create dataset with the construct_dataset.ipynb
4) Train U-Net with the train_U_Net.ipynb
5) Train EfficientNet with train_EfficientNet.ipynb
6) Calculate ROC curves with eval.ipynb


### Main Dependencies: 

The code was used on following libraries:

- tensorflow==2.4.0
- astra==2.1.0
- pydicom==2.3.0
- pandas==1.4.2

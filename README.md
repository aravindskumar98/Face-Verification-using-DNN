# Face-Verification-using-DNN
Pytorch implementation of face verification on LFW pairs dataset using embeddings obtained from VGG-16 and AlexNet

## Contents

This repository contains two notebooks that run out of the box. One corresponds to face verification using AlexNet and the second one uses VGG-16.

# Method

- Given the data, each instance consisted of two face images. The images were scaled and normalized for the VGG/AlexNet model. 
Then, two dataloaders were created (for each image in the pair).
- Initially, untrained AlexNet and VGG-16 models were evaluated on the test dataset to set a baseline. 
- From the two 4096x1 embeddings (obtained from corresponding images from both data loaders), the cosine Similarity was calculated. This is repeated for all images in the dataset
- The ROC curve and AUC was calculated using CosineSimilarity and target labels (matched:1, mismatched:0)
- The models are then fine tuned at very low learning rates (1e-6) for 30 epochs using Binary Cross Entropy loss between labels and cosine similarity of paired image embeddings.

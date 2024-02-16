Code for paper "A Comparative Study of Vision Transformer and Convolutional Neural Network Models in Fault Detection"

filename|usage
-------------|-------------
models|including 6 models:unet, resnet50unet,transunet transattunet,swinunet, and swindeeplay
dataset|FaultsegDataset:FaultSeg3D<br>ThebeDataset:Thebe<br>RandSelectThebe:random select Thebe
configs|config file of swin transformer
dataefficient|predit3DThebe.py: the process of using 2d model to predict 3D cube
losses| loss file
utils|utility of the project, including handle images, models and metrics
model_evaluation|evaluate_thebe.py/evaluate_faultseg:get metrics results<br>visual_Thebe/FaultSeg:visual predict fault lines<br>predict_kerry3D:predict kerry3D fault<br>predict3DThebe:predict Thebe and recover to 3D cube
warmup_schedule|warmup file 
optins.py|config file
train_Faultseg.py|train Faultseg
train_Thebe.py|train Thebe
                 
                 # Comparing-CNN-and-ViT-in-Geological-Fault-Detection
# Comparing-CNN-and-ViT-in-Geological-Fault-Detection

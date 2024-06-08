# Detecting and Classifying Neurofibromas using Deep Learning
# Tuong Hieu Huynh

The project codes include the following files: 
•	jsonfile_to_masks.ipynb
•	patching.ipynb
•	nf1_training.ipynb
•	nf1_validation.ipynb 
•	GUI_01.ipynb 
•	GUI_02.ipynb

These above project codes are uploaded to the GitHub link provided as below:
GitHub Link: https://github.com/tuonghieuhuynh/NF1_Project_Code.git

The trained model is uploaded to Google Drive:
•	"model" folder which contains “model_ep584.pth” file for trained model weights
•	Google Drive link: https://drive.google.com/drive/folders/1kUVZirMWN8d6rjbDwgvjmV46YdryDDMc?usp=sharing

Contents:
--------
•	jsonfile_to_masks.ipynb: This script automates the creation of mask images from annotated data, useful for tasks such as preparing image/mask dataset for deep learning in image segmentation. Make sure the correct file directory, file extension (json, png,..),database path, folder names... for use. 

•	patching.ipynb: This script automates the creation of smaller image patches from larger images and organizes them into designated training, validation, and testing folders, which is useful for deep learning and image processing tasks. Make sure the correct file directory, file extension (json, png,..),database path, folder names... for use.

•	nf1_training.ipynb:The code performs essential tasks for training and evaluating a U-Net model for image segmentation using PyTorch. It starts by importing necessary libraries and installing required packages. A SegmentationDataset class is defined to load, transform, and return images and masks as tensors, with data augmentation handled by albumentations. The U-Net model is constructed with specific layers for downsampling, upsampling, and convolutions. The training setup includes defining the loss function, optimizer, and evaluation metrics. The code then loads the training and validation datasets, creates data loaders, and iterates through training and validation loops to train the model and record metrics. It also saves the model's state at intervals and visualizes the training and validation losses over epochs using seaborn and matplotlib. The trained model with weights (“model_ep584.pth” file) will be saved for validation and testing purpose

•	nf1_validation.ipynb : The code evaluates a pre-trained U-Net model for image segmentation. It loads model weights, sets up the loss function and evaluation metrics, and then processes validation data to compute DICE and IOU scores. The evaluation involves selecting a random validation image, predicting its mask, and plotting the results alongside the ground truth. Finally, it calculates and prints the average DICE and IOU scores across the entire validation dataset, providing a concise assessment of the model's performance.

•	GUI_01.ipynb: The GUI1 allows users to upload an image (preferably with dimensions that are multiples of 256x256 for optimal precision, for example of sizes: 512x256 , 512x512, 768x512, ...). The code processes the uploaded image by splitting it into patches, transforming and predicting each patch, and then reconstructing the predicted mask. The predicted mask is displayed alongside the original image, and the NF1 severity is calculated and shown. Users can save the predicted mask image, and the GUI1 provides immediate visual feedback and NF1 severity information after saving. The entire interface is implemented using Tkinter for the GUI1, with OpenCV and PyTorch handling image processing and model inference.

•	GUI_02.ipynb: The provided code sets up GUI2, which, like GUI1, is designed for testing image segmentation using a pre-trained U-Net model. However, GUI2 includes additional functionality for displaying and plotting ground truth masks, making it suitable for thesis purposes. While GUI1 is focused on practical application, GUI2 enhances the presentation by showing the original image, the predicted mask, and the ground truth mask for comprehensive evaluation and analysis.

Support:
-------
For any queries contact,
Email: tuonghhuynh@gmail.com or tuonghieu.huynh@uq.net.au

Note:
----
This project is made available for research purposes only. 

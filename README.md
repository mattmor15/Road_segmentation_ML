
# Combining Boundary and Regional Loss Methods for Road Segmentation Tasks
## CS-433 Project 2. Challenge: Road Segmentation
In this repository we provide the required documentation for the submission of this project: run.ipynb and report.pdf.
## Code
In this section we explain how to proprerly setup and run the code. Please note that in order to run the code a GPU is required.
Note that our application, was run using Google Colab but can be run locally or using other platforms by changing the paths to your data. 
### Setup
#### Retrieving Data & Folder Organization
First you will need to download the dataset from the [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation) platform. We suggest uploading the data to a folder on Google Drive if you are running this from Colab. You will also need to create and redirect the paths to your predictiion (you will need one folder for masks before post-processing and one after post-processing), submission and logging folders. Run the first cells to install external libraries. 
#### Libraries
To run this notebook you will need Python 3.8 or later with the following libraries installed:
- `Pillow`
- `torch` > 2.1.0
- `torchvision` > 0.16.0
- `numpy`
- `matplotlib`
- `segmentation-models-pytorch`
- `pytorch-lightning`
- `wandb`
- `tensorflow`
- `scipy`
- `OpenCV`

For more details on the specific requirements, check the requirements.txt file.
### Preprocessing
In this section we prepare our datset to train the model. First we carry out data augmentation techniques to enhance the variability and the robustness of the data; in particular, we implemented random rotation, random cropping and resizing, photometric distorsions, and randonom flip. Then we convert the masks to tensors (torch.float32) and calclulate the distance maps from the edges of the classes, to obtain the distance map masks for boundary loss using the a custom `calc_dist` method inspired from the discussion forum of [Kervadec et al., 2019](https://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf)'s article and their [GitHub Repository](https://github.com/LIVIAETS/boundary-loss). Finally, we split the dataset for training the model and make validations of our predictions.
### Model
To run this section you will need a [wandb.ai](https://wandb.ai/site) account; you will need to input your Wandb API key in the appropriate cell: `wandb.login(key = "your key")`. 
First we define the model 'RoadSegmenter', we preprocess the images and implement costum-tailored loss functions (i.e. surface_loss, eval_surface_loss and eval_combined_loss) inspired again from [Kervadec et al., 2019](https://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf) and their [GitHub Repository](https://github.com/LIVIAETS/boundary-loss).
Then, we configure an Adam optimizer and a W&B (Weights & Biases) logger for monitoring the training. We use a callback to update the parameter 'alpha' used in the combined loss function. We save the model checkpoints and record the training on Weght&biases. Finally we train our model
### Predict on test set and save the result
Here, we apply our model to make the predictions on the test dataset. We save the segmented masks following the path in `save_dir`, initialized in the setup section. Then, we visualize our predictions, comparing them to the ground truth.
### Post processing 
In this section we use `OpenCV` library to apply a morphological transformation called opening. We take the segmented images from the directory where they are saved. Then, we define a kernel, which is a 3x3 matrix of ones, and we apply erosion and dilatation to the images. Finally, we save the post processed images following the path in `output_dir_post_proc`.
### Save for the submission
In this section, we prepare the post-processed masks for submission. The code divides the images into 16x16 patches, and for each one, assigns a label based on the average pixel values within that patch, with 1 for roads and 0 for background. Subsequently, we generate strings containing the image ID, the coordinates of the patch, and the corresponding label. Finally, these strings and labels are converted into a .csv file, which is saved in the folder following the path in `save_dir_fin` as initialized in the setup section.

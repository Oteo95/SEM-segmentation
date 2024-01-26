import logging
import numpy as np
import torch
from torch.optim import Adam
from transformers import SamModel, SamConfig, SamProcessor
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
from PIL import Image
import matplotlib.pyplot as plt

# Constants and Configuration
MODEL_CHECKPOINT = "facebook/sam-vit-base"
TRAIN_BATCH_SIZE = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
      self.dataset = dataset
      self.processor = processor

    def __len__(self):
      return len(self.dataset)

    def __getitem__(self, idx):
      item = self.dataset[idx]
      image = self.process_image(item["image"])
      ground_truth_mask = np.array(item["label"])
      # get bounding box prompt
      prompt = self.get_bounding_box(ground_truth_mask)

      # prepare image and prompt for the model
      inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

      # remove batch dimension which the processor adds by default
      inputs = {k:v.squeeze(0) for k,v in inputs.items()}

      # add ground truth segmentation
      inputs["ground_truth_mask"] = ground_truth_mask

      return inputs

    @staticmethod
    def process_image(image):

        # Resize image if necessary
        # Example: Resize to 224x224
        desired_size = (224, 224)
        if image.size != desired_size:
            image = image.resize(desired_size)

        return image
    
    @staticmethod
    def get_bounding_box(ground_truth_map: np.ndarray) -> list:
        """Calculate the bounding box from the mask."""
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = ground_truth_map.shape
        bbox = [
            max(0, x_min - np.random.randint(0, 20)),
            max(0, y_min - np.random.randint(0, 20)),
            min(W, x_max + np.random.randint(0, 20)),
            min(H, y_max + np.random.randint(0, 20))
        ]
        return bbox


class SAMTrainer:
    """Class to handle training of SAM model."""

    def __init__(self, dataset):
        self.processor = SamProcessor.from_pretrained(MODEL_CHECKPOINT)
        self.train_dataset = SAMDataset(dataset, self.processor)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
        
        self.model = SamModel.from_pretrained(MODEL_CHECKPOINT)
        for name, param in self.model.named_parameters():
            if "vision_encoder" in name or "prompt_encoder" in name:
                param.requires_grad_(False)

        self.optimizer = Adam(self.model.mask_decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def train(self):
        self.model.to(DEVICE)
        self.model.train()
        for epoch in range(NUM_EPOCHS):
            epoch_losses = []
            for batch in tqdm(self.train_dataloader):
                self._train_batch(batch, epoch_losses)
            print(f'EPOCH: {epoch}\nMean loss: {mean(epoch_losses)}')

    def _train_batch(self, batch, epoch_losses):
        outputs = self.model(pixel_values=batch["pixel_values"].to(DEVICE),
                             input_boxes=batch["input_boxes"].to(DEVICE),
                             multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(DEVICE)
        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        epoch_losses.append(loss.item())
    
    def load_model(self, local_path: str):
        # Load the model configuration
        self.model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

        # Create an instance of the model architecture with the loaded configuration
        self.model = SamModel(config=self.model_config)
        #Update the model by loading the weights from saved file.
        self.model.load_state_dict(torch.load(
                local_path
            )
        )
        
        # set the device to cuda if available, otherwise use cpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
    
    def inference(self, patch):

        # Define the size of your array
        array_size = 256

        # Define the size of your grid
        grid_size = 10

        # Generate the grid points
        x = np.linspace(0, array_size-1, grid_size)
        y = np.linspace(0, array_size-1, grid_size)

        # Generate a grid of coordinates
        xv, yv = np.meshgrid(x, y)

        # Convert the numpy arrays to lists
        xv_list = xv.tolist()
        yv_list = yv.tolist()

        # Combine the x and y coordinates into a list of list of lists
        input_points = [[[int(x), int(y)] for x, y in zip(x_row, y_row)] for x_row, y_row in zip(xv_list, yv_list)]

        #We need to reshape our nxn grid to the expected shape of the input_points tensor
        # (batch_size, point_batch_size, num_points_per_image, 2),
        # where the last dimension of 2 represents the x and y coordinates of each point.
        #batch_size: The number of images you're processing at once.
        #point_batch_size: The number of point sets you have for each image.
        #num_points_per_image: The number of points in each set.
        input_points = torch.tensor(input_points).view(1, 1, grid_size*grid_size, 2)
        
        single_patch = Image.fromarray(patch)
        # prepare image for the model

        #First try without providing any prompt (no bounding box or input_points)
        #inputs = processor(single_patch, return_tensors="pt")
        #Now try with bounding boxes. Remember to uncomment.
        inputs = self.processor(single_patch, input_points=input_points, return_tensors="pt")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Move the input tensor to the GPU if it's not already there
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.eval()


        # forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        # apply sigmoid
        single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
        single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)

        plt.imshow(single_patch_prob)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first image on the left
        axes[0].imshow(np.array(single_patch), cmap='gray')  # Assuming the first image is grayscale
        axes[0].set_title("Image")

        # Plot the second image on the right
        axes[1].imshow(single_patch_prob)  # Assuming the second image is grayscale
        axes[1].set_title("Probability Map")

        # Plot the second image on the right
        axes[2].imshow(single_patch_prediction, cmap='gray')  # Assuming the second image is grayscale
        axes[2].set_title("Prediction")

        # Hide axis ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Display the images side by side
        plt.show()
        
        return outputs
     
            

if __name__ == "__main__":
    # Usage Example
    # dataset needs to be defined or loaded before this
    trainer = SAMTrainer(dataset)
    trainer.train()

    # Save the trained model
    torch.save(trainer.model.state_dict(), "/workspaces/SEM-segmentation/src/weights/sam/mito_model_checkpoint.pth")
    

# Repository for Masters Thesis Project

### Short description

This project will focus on developing and training DL models, specifically CNNs, to predict the species composition of vascular plant assemblages in Norway using satellite-derived climatic data. We will use species occurrence records from the Global Biodiversity Information Facility (GBIF) and associate them with high-resolution climatic variables from sources like CHELSA [(Karger et al., 2021)](https://www.zotero.org/google-docs/?bLQSd8).

### Main Objectives:

#### 1. Develop Deep Learning Models for Species Composition Prediction

- Implement and train CNNs to predict vascular plant assemblage composition in Norway using satellite-derived climate data.
    
- Optimize the model architecture to accurately capture the relationship between climatic variables and species presence across different environmental conditions.
    

#### 2. Enhance Understanding of Climatic Drivers on Plant Assemblages

- Use interpretability techniques to identify and analyze the most influential climatic factors shaping plant assemblages.
    
- Quantify how different climate variables contribute to species composition and assess their ecological significance.
    

#### 3. Predict Species Composition in Undersampled and Future Scenarios

- Utilize the trained models to estimate species compositions in poorly sampled regions of Norway.
    
- Apply future climate projections to forecast changes in vascular plant assemblages under different climate change scenarios.![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXfXyCDU5eSdewtWPVPSsQn1I9v7t6puA6U3_CVmqZb82UcCRUEhKGCgBOT-1WEpmqY3BR4qmhNQi25wVvsnP_fopSz0R80nEH3cPLA_Tg5PEj-r4WFymvgG5avomDAf0XMx-ol5?key=RmwAfgQCK3ztacxMBzsRXSzW)
    

**

### Using CNN applied to Multi-label classification problems

Building a Species Distribution Model (SDM) that interprets climatic maps to predict  vegetation plots compsition can be done by applying [Convolutional Neural Networks (CNN)](https://medium.com/thedeephub/convolutional-neural-networks-a-comprehensive-guide-5cc0b5eae175) to a [multilabel-classification](https://medium.com/data-science-in-your-pocket/multi-label-classification-for-beginners-with-codes-6b098cc76f99). 

Vegetation plots can be encoded as binary vectors where 0 corresponds to absence and 1 corresponds to presence. Thus if we have encoded in a fixed position all the 1819 plant species in norway, we can represent the present and absent species in a determinated area using a binary vector of 1819 variables.

**Species composition vector example:**

![[Pasted image 20250403103202.png]]
![[Pasted image 20250403103343.png]]

As you can see, we have multiple 0 (absent species) and multiple 1 (present species). 
In the same vegetation plot we can have many present species (many 1 values) in the species composition vector.
In a multilabel classification we can have many correct instances (present species) for a single sample. So this kind of classification approach perfectly suits our problem!!

### CNN-SDM model diagram

![[Pasted image 20250403103853.png]]

### Data Pipeline
#### Plant assemblage Dataset

I used an [Rscript](https://github.com/rauletepawa/Species_Distribution_Modeling/blob/main/code/1_gbif_norge_data.R) to download all the GBIF occurrences in vascular plants in Norway main land from 1991 to 2020. I applied some filters to remove all those occurrences located in capitals, institutions, seas or that did not have any valid coordinates assigned inside Norway.

Then I obtained a occurrence clean dataset with a total of 3.011.729 occurences.
Using the [splitting norway in grids](https://github.com/rauletepawa/Species_Distribution_Modeling/blob/main/code/Splitting_Norway_in_grids.ipynb) script we splitted norway in grids of 1km and I counted as a co-occurrence all the species that are observed in the same grid at the same year. Thus I built a plant assemblage dataset that includes all the plant species that co-occur in the same 1km grid the same year filtering all those assemblages that including less than 5 co-occurrences.

**Here there is an example of the dataset:**
![[Pasted image 20250403122559.png]]

![[Pasted image 20250403122302.png]]

#### Climatic Dataset
The climatic dataset is composed by a total of 59.074 plant assemblages collected from 1991 until 2018. For each vegetation plot coordinates (location) I extracted a 11 channels (variables) 32x32 climatic map at 1km resolution (1 pixel corresponds to 1km).
This climatic dataset construction can be followed in the [Build CNN dataset](https://github.com/rauletepawa/Species_Distribution_Modeling/blob/main/code/Build_CNN_dataset.ipynb) script.

We also transformed the plant assamblages into binary presence/absence vectors with the [filtering species script](http://localhost:8888/notebooks/Projects/GitHub/Species_Distribution_Modeling/code/filtering_species.ipynb) 

Here you can see how the final CNN dataset looks like:

![[Pasted image 20250403122951.png]]

#### Climatic Variables

The 11 variables stacked in the climatic maps, acoording to CHELSA's manual, correspond to:

- bio01d: mean annual air temperature, mean annual daily mean air temperatures averaged over 1 year
- bio04d: temperature seasonality, standard deviation of the montlhly temperatures
- bio12d: annual precipitation, accumulated preccipitation amount over 1 year
- cdd: consecutive_dry_days_index_per_time_period, number_of_cdd_periods_with_more_than_5days_per_time_period
- fd: frost_days_index_per_time_period
- gdd5: Growing degree days heat sum above 5°C, heat sum of all days above the 5°C temperature accumulated over 1 year
- prsd: precipitation
- scd: Snow cover days, Number of days with snowcover calculated using the snowpack model implementation in from TREELIM
- swe: Snow water equivalent, Amount of liquid water if snow is melted
### Building the CNN model

#### Loading the training data:

To build the CNN model both [Tensorflow]() and [Pytorch]() have been tested. Resulting Pytorch the best framework to asses this kind of problem as there are more resources online to costumize specific metrics for evaluating the plant assemblage prediction models. 

However, both pipelines in pytorch and tensorflow are uploaded in this repository.

The main file for the model building is the [TFM_CNN_pytorch](https://github.com/rauletepawa/Species_Distribution_Modeling/blob/main/code/TFM_CNN_pytorch.ipynb) file. There I have programmed a reproducible training pipeline where I load the final climatic assemblages dataset in a pytorch data loader with the **MultiLabelDataset** function:

````
from sklearn.model_selection import train_test_split

In [13]:

class MultiLabelDataset(Dataset):
    def __init__(self):
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = None, None, None, None, None, None
        self.mode = 'train'

        self.images_no_perm = torch.tensor(np.stack(df['climatic_map'].values).astype(np.float32))#.to(device)  # Move data to GPU
        self.labels = torch.tensor(np.stack(df['species_vector'].values).astype(np.float32))#.to(device)  # Move labels to GPU
        self.images = []
    
        permuted_images = []
        for image in self.images_no_perm:
            permuted_image = image.permute(2, 0, 1)
            self.images.append(permuted_image)
        
        

    def train_val_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.images, self.labels, test_size = 0.15, random_state=42)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size = 0.15, random_state=42)
    def __len__(self):
        if self.mode == 'train':
            return len(self.x_train)
        elif self.mode == 'val':
            return len(self.x_val)
        elif self.mode == 'test':
            return len(self.x_test)

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample = {'images': self.x_train[idx], 'labels': self.y_train[idx]}
        elif self.mode == 'val':
            sample = {'images': self.x_val[idx], 'labels': self.y_val[idx]}
        elif self.mode == 'test':
            sample = {'images': self.x_test[idx], 'labels': self.y_test[idx]}
        return sample
````

#### Training Loop Pytorch:

In order to train the models efficiently I coded this training loop function that takes the dataloader,  dataset,  epochs, model type, custom loss and optimizer as variables that we can change and adjust acoordingly

```
def model_training(dataloader,dataset,num_epochs,model,custom_loss,optimizer):
    
    # dataloader = DataLoader(dataset, batch_size = 32, shuffle = False)

    epoch_train_loss = []
    epoch_val_loss = []

    for epoch in range(num_epochs):
        train_losses = []
        dataset.mode = 'train'
        model.train()
        for D in dataloader:
            optimizer.zero_grad()
            data = D['images'].to(device, dtype = torch.float)
            labels = D['labels'].to(device, dtype = torch.float)
            y_hat = model(data)
            loss = custom_loss(y_hat, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        epoch_train_loss.append(np.mean(train_losses))

        val_losses = []
        val_iou = []

        dataset.mode = 'val'
        model.eval()
        with torch.no_grad():
            for D in dataloader:
                data = D['images'].to(device, dtype = torch.float)
                labels = D['labels'].to(device, dtype = torch.float)
                y_hat = model(data)
                loss = custom_loss(y_hat, labels)
                val_losses.append(loss.item())

        epoch_val_loss.append(np.mean(val_losses))

        print(f'Train Epoch: {epoch+1} \t Train Loss:{np.mean(train_losses)} \t Val Loss: {np.mean(val_losses)}')
    
    return model, epoch_train_loss, epoch_val_loss
```
#### Establishing a Baseline:

In order to be able to set experiments and try different loss functions and evaluation metrics I architected a simple CNN to establish a baseline and determine if its possible to train a model from scratch on climatic maps to predict species assemblage composition.

```
class CNNModel_1(nn.Module):
    def __init__(self, input_channels,input_size, num_classes):
        super(CNNModel_1, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, padding=2)  # 'same' padding in TensorFlow = padding=2
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling2D(pool_size=(2, 2))

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling2D(pool_size=(2, 2))

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling2D(pool_size=(2, 2))
        
        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling2D(pool_size=(2, 2))

        # Fully Connected Layers
        self.fc1 = nn.Linear(512 * (input_size // 16) * (input_size // 16), 1024)  # Flatten output size depends on input size
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        x = torch.flatten(x, start_dim=1)  # Flatten the output for fully connected layers

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)   # x = torch.sigmoid(self.fc3(x))  # Assuming binary classification

        return x
```
#### Loss Functions:

To make the baseline CNN converge in the most efficient way possible I made several experiments with 3 loss metrics:
- [Binary Cross Entropy (BCE)](https://medium.com/data-science/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
- [Focal Loss](https://arxiv.org/pdf/1708.02002) 
- [Dice Loss](https://cvinvolution.medium.com/dice-loss-in-medical-image-segmentation-d0e476eb486) 

Also BCE and Focal Loss where tested with weighting and without weighting. And I have chose weighting as it has shown better convergence plots such as it is described in the [original paper](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14466?utm=) where the CustomLoss class is described:

```
import torch.nn.functional as F
import torch.distributed as dist
import argparse

class CustomLoss(nn.Module):
    def __init__(self, loss_fn, args):
        super(CustomLoss, self).__init__()
        self.loss_fn = loss_fn
        self.weighted = args.weighted
        self.args = args
        if self.weighted:
            self.samples_per_cls = list(args.train_label_cnt.values())
            self.no_of_classes = args.num_classes

    def compute_weights(self, labels, beta=0.9999):
        effective_num = 1.0 - np.power(beta, self.samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * self.no_of_classes
        labels_one_hot = labels

        weights = torch.tensor(weights).float().to(self.args.device, non_blocking=True)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        
        return weights
    
    def forward_focal(self, logits, labels,alpha=0.999,gamma=2.0):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if self.weighted:
            weights = self.compute_weights(labels)
            weights_t = weights * labels * alpha + (1 - labels) * (1 - alpha)
            weighted_loss = weights_t * loss
            focal_loss = weighted_loss.mean()
        
        else:
            alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
            focal_loss = alpha_t * loss
            focal_loss = focal_loss.mean()
        
        return focal_loss

    
    def forward_bce(self, logits, labels):
        if self.weighted:
            weights = self.compute_weights(labels)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight = weights)
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return bce_loss
    
    
    def forward_dice(self, logits, labels):
        p = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (p * labels).sum(0)
        total = (p**2 + labels**2).sum(0)
        dice_loss = 1 - (intersection + smooth)/(total + smooth)
        
        return dice_loss.mean()
    
    
    def forward(self, logits, labels):
        if self.loss_fn == 'bce':
            return self.forward_bce(logits, labels)
        elif self.loss_fn == 'focal':
            return self.forward_focal(logits, labels)
        elif self.loss_fn == 'dice':
            return self.forward_dice(logits, labels)
        
# Example args (you can replace it with your actual arguments)
class Args:
    def __init__(self):
        self.weighted = True  # Set to False if no class weighting is needed
        self.train_label_cnt = class_counts_dict  # Example class counts
        self.num_classes = 1819  # Binary classification
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = 'CNN'
        self.eval = False
        self.thres_method = 'adaptive' # 'global'
        self.threshold = 0.15
```

#### Model Evaluation:

The CNN model outputs a logits vector for each training sample, after applying a [sigmoid](https://machinelearningmastery.com/a-gentle-introduction-to-sigmoid-function/) function to these logits vectors, we obtain a probability vector where each probability value is independent of the other probabilities (they do not add up to 1 as in the softmax function). 

##### BCE loss training
![[Pasted image 20250403135323.png]]
##### FOCAL loss Training

![[Pasted image 20250403135641.png]]
#### ResNet FineTunning:





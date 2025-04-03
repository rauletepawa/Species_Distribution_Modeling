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

[](https://github.com/rauletepawa/Species_Distribution_Modeling/blob/main/Images/Pasted%20image%2020250403103202.png)
[](https://github.com/rauletepawa/Species_Distribution_Modeling/blob/main/Images/Pasted%20image%2020250403103328.png)

As you can see, we have multiple 0 (absent species) and multiple 1 (present species). 
In the same vegetation plot we can have many present species (many 1 values) in the species composition vector.
In a multilabel classification we can have many correct instances (present species) for a single sample. So this kind of classification approach perfectly suits our problem!!

### CNN-SDM model diagram

![[Pasted image 20250403103853.png]]

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







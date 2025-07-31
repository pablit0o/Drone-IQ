# Drone-IQ
Repository for the Autonomous Drone project from COSMOS 2024-2025 Cluster 2.

CODE ARCHITECTURE NOTES:
- Not all files are required for the processing of images. Look into main.py to check which ones are necessary.
- It is important you download all of the libraries found throughout the .py files
- You might need to setup an environment with tensorflow in order to use our files. (Python 3.11 or below)
- We split up the code files into three main sections, but it is fine if you have it all under one comprehensive environment.
- Our 207 image dataset, including the json file, is included down below:
- www.kaggle.com/datasets/nnishio/building-outlines-for-segmentation
- Aside from just downloading libraries, you MUST change your file paths in main.py.

PROJECT DESCRIPTION:

The goal of our project was to determine the height of various buildings around the campus of UCI using simple drone images. We would require firstly a model that could accurately determine the outlines of buildings from a drone image, and also a model that could implement the parallax method to determine the height. Firstly, we collected a dataset of 207 images to train a DeepLabV3 segmentation model on. The dataset can be found here. Then, we pre-processed the images by applying various transformations, and augmented the images as well. After that, we trained the DeepLabV3 model on the 207dsf images. We eventually achieved a peak Dice Coefficient of roughly 96% for the non-processed images. 

After this, we took drone images and created prediction masks. We used a depth-first search algorithm to clean the predicted masks to create a single blob from which we could extract the lowest and highest white pixel. From this, we were able to use the parallax method to determine the relevant angles of the triangles from two subsequent images. By knowing the vertical displacement of the drone, we were able to determine the angles and heights of the buildings (for more info on how this was done view the poster). In the end, comparing the parallax heights to the real height, we had an average percent error of 22.84%. We also determined the model’s accuracy as when testing multiple instances of the model on the same building (Engineering Tower), the mean of the sample fell within one standard deviation of the true result, meaning the model was relatively accurate.


DATASET:
- The dataset that was used to train this model can be accessed at https://www.kaggle.com/datasets/nnishio/building-outlines-for-segmentation/data
  
POSTER:
- The poster for this project is below:
- [COSMOS Cluster 2 Drones Poster.pdf](https://github.com/user-attachments/files/21535167/COSMOS.Cluster.2.Drones.Poster.pdf)

FUTURE CHALLENGES:
- In case there are any questions or challenges you have, feel free to shoot us an email.
- Nelson Nishio: nelsonknishio@gmail.com
- Pablo Silva: pbsilva2009@gmail.com
- Arjun Maganti: arjunmaganti2008@gmail.com
- Amun Reddy: amunreddy@gmail.com

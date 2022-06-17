# Fashion Recommender System

##  Proposed methodology

In this project, we propose a model that uses Convolutional Neural Network and the Nearest 
neighbour backed recommender. Initially, the neural networks are trained and then 
an inventory is selected for generating recommendations and a database is created for the items in 
inventory. The nearest neighbour’s algorithm is used to find the most relevant products based on the 
input image and recommendations are generated.

## Training the Neural Networks

Once the data is pre-processed, the neural networks are trained, utilizing transfer learning 
from ResNet50. More additional layers are added in the last layers that replace the architecture and 
weights from ResNet50 in order to fine-tune the network model to serve the current issue. The figure
shows the ResNet50 architecture.

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/72528f2b4197cc5010227068ec72cd10f71214d4/Demo/resnet.png?raw=true "Face-Recognition-Attendance-System")

## Getting the Inventory

The images from Kaggle Fashion Product Images Dataset. The 
inventory is then run through the neural networks to classify and generate embeddings and the output 
is then used to generate recommendations. The Figure shows a sample set of inventory data

![Alt text](https://github.com/sonu275981/Clothing-recommender-system/blob/1e51a0d1db0e171e8d496524aa95a0098241fb1b/Demo/inventry.png?raw=true "Face-Recognition-Attendance-System")

## Recommendation Generation

To generate recommendations, our proposed approach uses Sklearn Nearest neighbours Oh Yeah. This allows us to find the nearest neighbours for the 
given input image. The similarity measure used in this Project is the Cosine Similarity measure. The top 5 
recommendations are extracted from the database and their images are displayed.

## Experiment and Results

The concept of Transfer learning is used to overcome the issues of the small size Fashion dataset. 
Therefore we pre-train the classification models on the DeepFashion dataset that consists of 44,441
garment images. The networks are trained and validated on the dataset taken. The training results 
show a great accuracy of the model with low error, loss and good f-score.

## Dataset Link

[Kaggle Dataset Big size 15 GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Kaggle Dataset Small size 572 MB](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)


## Built With

- [OpenCV]() - Open Source Computer Vision and Machine Learning software library
- [Tensorflow]() - TensorFlow is an end-to-end open source platform for machine learning.
- [Tqdm]() - tqdm is a Python library that allows you to output a smart progress bar by wrapping around any iterable.
- [streamlit]() - Streamlit is an open-source app framework for Machine Learning and Data Science teams. Create beautiful data apps in hours, not weeks.
- [pandas]() - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- [Pillow]() - PIL is the Python Imaging Library by Fredrik Lundh and Contributors.
- [scikit-learn]() - Scikit-learn is a free software machine learning library for the Python programming language.
- [opencv-python]() - OpenCV is a huge open-source library for computer vision, machine learning, and image processing.

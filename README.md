### Artist Prediction from Artworks by Fully Retraining Pretrained Convolutional Neural Networks (InceptionResNetV3)

<pre>
Domain             : Computer Vision, Machine Learning
Sub-Domain         : Deep Learning, Image Recognition
Techniques         : Deep Convolutional Neural Network, Transfer Learning, InceptionResNetV3
Application        : Image Recognition, Image Classification, Art
</pre>

### Description
1. Detected Artists from their Artworks with Deep Learning (Convolutional Neural Network) specifically by retraining pretrained model "InceptionResNetV3" completely from scratch.
2. Before feeding data into model, preprocessed and augmented image dataset containing 8,446 images (2GB) by adding random horizontal flips, rotations and width and height shifts.
3. After loading pretrained model "InceptionResNetV3", added global average pooling 2D with and dense layer with 512 units followed by batch normalization, dropout layers for regularization and activation for only dense layer. Finally, added final output layer - a dense layer with softmax activation and compiled with optimizer-Adam with learning rate-0.0001, metric-accuracy and loss-categorical cross-entropy.
4. Trained for 15 iterations and attained training accuracy 98.36% and loss (categorical cross-entropy) 0.0820 and validation accuracy of 78.75% and loss 0.9093.

#### Code
<pre>
GitHub Link      : <a href=https://github.com/anjanatiha/Artist-Prediction-from-Artworks/blob/master/Artist%20Prediction%20from%20Artworks.ipynb>Artist Prediction from Artworks (GitHub)</a>
GitLab Link      : <a href=https://gitlab.com/anjanatiha>Artist Prediction from Artworks (GitLab)</a>
Kaggle Kernel    : <a href=https://www.kaggle.com/anjanatiha/artist-prediction-from-artworks?scriptVersionId=13206427>Artist Prediction from Artworks</a>
Portfolio        : <a href=https://anjanatiha.wixsite.com/website>Anjana Tiha's Portfolio</a>
</pre>

#### Relevant Papers
##### 1.
<pre>
Link             : <a href=http://cs231n.stanford.edu/reports/2017/pdfs/406.pdf>Artist Identification with Convolutional Neural Networks</a>

@article{viswanathan2017artist,
  title={Artist Identification with Convolutional Neural Networks},
  author={Viswanathan, Nitin},
  journal={transfer},
  volume={77},
  pages={89--8},
  year={2017}
}
</pre>

##### 2.
<pre>
Link             : <a href=http://www.vcl.fer.hr/papers_pdf/Fine-tuning%20Convolutional%20Neural%20Networks%20for%20fine%20art%20classification.pdf>Fine-tuning Convolutional Neural Networks for fine art classification</a>

@article{cetinic2018fine,
  title={Fine-tuning Convolutional Neural Networks for fine art classification},
  author={Cetinic, Eva and Lipic, Tomislav and Grgic, Sonja},
  journal={Expert Systems with Applications},
  volume={114},
  pages={107--118},
  year={2018},
  publisher={Elsevier}
}
</pre>

#### Dataset
<pre>
Dataset Name     : Best Artworks of All Time
Dataset Link     : <a href=https://www.kaggle.com/ikarus777/best-artworks-of-all-time>Best Artworks of All Time (Kaggle)</a>
<!--
Original Dataset : <a href=https://ceb.nlm.nih.gov/repositories/malaria-datasets/>Malaria Datasets - National Institutes of Health (NIH)
-->
</a>
</pre>

### Dataset Details
<pre>
Dataset Name            : Best Artworks of All Time
Number of Class         : 50
</pre>

| Dataset Subtype | Number of Image | Size of Images (GB/Gigabyte) |
| :-------------- | --------------: | ---------------------------: |
| **Total**       | 8,446           | 2 GB                         |
| **Training**    | 6,357           | ---                          |
| **Validation**  | 2,089           | ---                          |
| **Testing**     |                 | ---                          |



### Model and Training Prameters
| Current Parameters   | Value                                                       |
| :------------------- | :---------------------------------------------------------- |
| **Base Model**       | InceptionResNetV3                                           |
| **Optimizers**       | Adam                                                        |
| **Loss Function**    | Categorical Crossentropy                                    |
| **Learning Rate**    | 0.0001                                                      |
| **Batch Size**       | 32                                                          |                                     
| **Number of Epochs** | 15                                                          |
| **Training Time**    | 1 hours 19 Min                                              |


### Model Performance Metrics (Prediction/ Recognition / Classification)
| Dataset              | Training       | Validation    | Test      |                                 
|:---------------------|---------------:|--------------:| ---------:|
| **Accuracy**         | 98.36%         | 78.75%        | ---       |
| **Loss**             | 0.0820         | 0.9093        | ---       |
| **Precision**        | ---            | ---           | ---       |
| **Recall**           | ---            | ---           | ---       |
| **Roc-Auc**          | ---            | ---           | ---       |


### Other Experimented Model and Training Prameters
| Parameters (Experimented) | Value                                                  |
|:--------------------------|:------------------------------------------------------ |
| **Base Models**           | InceptionResNetV3                                      |
| **Optimizers**            | Adam                                                   |
| **Loss Function**         | Categorical Crossentropy                               |
| **Learning Rate**         | 0.001, 0.0001                                          |
| **Batch Size**            | 32, 64, 128                                            |                                     
| **Number of Epochs**      | 10, 15, 100                                            |
| **Training Time**         | 1 hours 19 Min                                         |

#### Tools / Libraries
<pre>
Languages               : Python
Tools/IDE               : Kaggle
Libraries               : Keras, TensorFlow, InceptionResNetV3
</pre>

#### Dates
<pre>
Duration                : March 2019 - Sep 2019
Current Version         : v1.0.0.10
Last Update             : 09.05.2019
</pre>

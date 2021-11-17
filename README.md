# Hello Siamese Network

## Siamese Network
- Contains two or more identical subnetworks used to generate feature vectors for each input and compare them
- Loss function: binary crossentropy (work but usually not effective), triplet function, contrastive function, mean square error, etc.
- Application: duplicate detection, anomalies detection, face recognition, etc.


## Experiment
Training procedure:
+ Step 1: Generate training set from MNIST. This training set consists of triples. A triple is defined as <b>(image 1, image 2, its similarity)</b>. If image 1 and image 2 are similar (based on their labels), their similarity is assigned to 1. Otherwise, their similarity is assigned to 0.

I also generate a test set in the same way.

In total, the training set has 100k triples. The test set has 50k triples. The training set and test set are balanced.

The source code of this step is provided in dataset_generation.py. Set the path of TRAINING_SET, TEST_SET correctly. The path of the training set and the test set are represented by MYTRAINING, MYTEST, respectively.

+ Step 2: Train on 70k triples. Evaluate the remaining 30k triples. There are 40 epochs. Of course, the number of epochs could be larger. However, because I am a lazy person, I chose this epoch based on my intuition.

+ Step 3: Choose the model achieving the highest accuracy on the validation set.

+ Step 4: Compute the accuracy of the model on the whole training set and test set.

The accuracy of the models could be better. I would run more epochs in the future.

### MNIST 

| Model  | Loss | Training set | Test set | Result 
| --- | --- | --- | --- |  --- |
| M1 (main1.py) | binary_crossentropy  | 0.98794 |	0.9539 | <a href="https://github.com/ducanhnguyen/siamese_network/tree/main/model/mnist/v1">link</a> |
| M2 (main2.py) | mse  | 0.99195|	0.9734 |  <a href="https://github.com/ducanhnguyen/siamese_network/tree/main/model/mnist/v2">link</a> |
| M3 (main3.py)  | contrastive loss  | 0.99069 |	0.9751 |  <a href="https://github.com/ducanhnguyen/siamese_network/tree/main/model/mnist/v3">link</a> |



### Fashion-MNIST 

| Model  | Loss | Training set | Test set | Result
| --- | --- | --- | --- | --- |
| F1 (main1.py) | binary_crossentropy  | 0.93579 |	0.91416 |  <a href="https://github.com/ducanhnguyen/siamese_network/tree/main/model/fashion-mnist/v1">link</a> |
| F2 (main2.py) | mse  | 0.93968	| 0.91858 | <a href="https://github.com/ducanhnguyen/siamese_network/tree/main/model/fashion-mnist/v2">link</a> |
| F3 (main3.py)  | contrastive loss  | 0.94351 |	0.9107 | <a href="https://github.com/ducanhnguyen/siamese_network/tree/main/model/fashion-mnist/v3">link</a> |

### How to use the trained models?
Just fed two images into the trained models. The output is the probability. The higher probability, the more similarity between the two images.

For example, I display 5 comparisons as follows:
![alt text](https://github.com/ducanhnguyen/siamese_network/blob/main/model/mnist/v1/sample.png)

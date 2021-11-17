# Hello siamese_network

## Siamese Network
- Contains two or more identical subnetworks used to generate feature vectors for each input and compare them
- Loss function: binary crossentropy (work but usually not effective), triplet function, contrastive function, mean square error, etc.
- Application: duplicate detection, anomalies detection, face recognition, etc.


## Experiment
Training procedure:
+ Step 1: Generate training set from MNIST. This training set consists of triple. A triple is defined as <b>(image 1, image 2, its similarity)</b>. If image 1 and image 2 are similar (based on their labels), theirs similarity is assigned to 1. Otherwise, theirs similarity is assigned to 0.

In total, there are 100k triples. Half of them are classifed as <b>similar</b>.
+ Step 2: Train on 70k triples. Evaluate on the remaining 30k triples.

+ Step 3: Choost the model achieving the highest accuracy on the validation set.

+ Step 4: Report the model on the whole training set and test set.

### MNIST 

| Model  | Loss | Training set | Test set
| --- | --- | --- | --- |
| M1  | binary_crossentropy  | 0.98794 |	0.9539 |
| M2  | mse  | 0.99195|	0.9734 |
| M3  | contrastive loss  | 0.99069 |	0.9751 |


### Fashion-MNIST 

| Model  | Loss | Training set | Test set
| --- | --- | --- | --- |
| F1  | binary_crossentropy  | 0.93579 |	0.91416 |
| F2  | mse  | 0.93968	| 0.91858 |
| F3  | contrastive loss  | 0.94351 |	0.9107 |

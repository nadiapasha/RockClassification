# Finetuning google/vit-base-patch16-224 model to classify images of different types of rocks
Only last linear layer was trained becuase of the shortage of training data. The accuracy on test data was ~50% for 7 classes. There were approximately 30 samples in each class.  
Learning Rate - 0.001
Batch Size - 2
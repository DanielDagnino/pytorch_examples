# Pytorch examples
Here I leave some examples that maybe are useful for someone who is starting learning Pytorch.

For a jupyter example of the 4 classes classifier you can visit https://www.kaggle.com/danieldagnino/training-a-classifier-with-pytorch

---
With these codes I pretend to show the common steps that we have to follow to write a program using Pytorch, independently if it is to build a simple neural network of a very complex one. The steps that I describe below are the following:

  1. Build the Dataset. We are going to generate a simple data set and then we will read it.
  2. Build the DataLoader.
  3. Build the model.
  4. Define the loss function and the optimizer.
  5. Train the model.
  6. Generate predictions.
  7. Plot the results. I hope it can be useful for someone who is starting programming using Pytorch.

<table>
  <tr>
    <td colspan="2"> These are the solutions obtained after running the two provided examples. </td>
  </tr>
    <td width="50%"> <img src="https://github.com/DanielDagnino/pytorch_examples/blob/master/img/2%20class.png" alt="Fianl circuit" width="400" /> </td>
    <td width="100%"> <img src="https://github.com/DanielDagnino/pytorch_examples/blob/master/img/4%20classes.png" alt="Valve" rotate="90" width="50%" /> </td>
  <tr>
    <td width="50%"> Figure 1: Binary classifier. Color boxes indicate the predictions of the model after the training and the circles the ground truth labels. Blue line represents the straight line obtained using the values of the linear layer of the model. </td>
    <td width="50%"> Figure 2: Classifier with 4 classes. Color boxes indicate the predictions of the model after the training and the circles the ground truth labels. </td>
  </tr>
</table>

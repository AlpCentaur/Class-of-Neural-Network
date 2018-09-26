# Class-of-Neural-Network
Optimizing the topology of a DNN based on functional analysis


Changes in the script DNN_anydim.py:

  Topology T4 is modified to work for any dimension.
  An attempt of writing a convolutional neural network (R-CNN).
  The attempt fails, as I am taking to many pixels and the computation gets too expensive. 
  After further research I found out that the working and spread approach takes only several pixels of every image (and not ALL!).  
  With this approach, it is possible to enlarge the Region of Interest(ROI). 
  Convolutional features are extracted through preprocessing in the class DNN.Get_Image_TrainData() (compute gradients, get angles) 


# Fake_News_Detection
The project is to detect fake from the fact news in the web-news data set. The results are confirming that the words with uncertainty will exist more frequent in the Fake news. Therefore, it is a classification problem.
Three machine learning algorithms have been implemented for the classification — the machine learning algorithms including, logistic regression, K-Nearest Neighbor (K-NN), and Random Forest. Moreover, the implemented deep learning models are RNN and GRU which are well-known algorithms and performing pretty good for neural networks particularly in sequential data. Deep learning models are implemented using Keras on top of TensorFlow.
The results show that:
-	Machine learning algorithms all are performing pretty good with the accuracy above 85%, and they are performing good as RNN does,
-	The best learning method is not necessarily the most complicated
-	The overfitting is negligible; however, in general it is higher in RF,
-	More Epoch is required to get better results in GRU
-	GRU model requires a long period of time to be implemented even in the GPU

# Wake word detection
This notebook runs through the steps needed to implement a wake-word detector. A wake word detector is used to trigger an event in a program upon the utterance of a specific word. The method with which this is achieved here is using a deep recurrent neural network based on gated recurrent units that takes as an input an audio clip of 10 seconds and returns an audio clip which rings a bell if the wake word was uttered. 

This notebook makes use of the model and pre-trained weights from the coursera course on sequential models. 

<img src="images/spectrogram.png?raw=true" width="400">
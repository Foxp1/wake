# Wake word detection
This notebook implements a wake-word detector. A wake word detector is used to trigger an event in a program upon the utterance of a wake word (similar to "Hey Siri" in Apple devices and "OK Google" in Google devices). It uses a deep recurrent neural network based on gated recurrent units. The training and development data is synthesized from audio samples (of backgrounds, wake words, and negatives), which can easily be generated using the recording section in this notebook. The pre-trained weights are then loaded and the model is fine tuned to the newly added audio samples.

This notebook uses the model and pre-trained weights from the [Coursera](https://www.coursera.org/) course on sequence models.  

As an example see the following figure. It shows the spectrogram of a sequence of negative words (Green, Purple, Yellow) and the wake word (Activate), and the model's prediction of when the wake word is said (for signals above the red horizontal line). 

<img src="images/spectrogram.png?raw=true" width="400">
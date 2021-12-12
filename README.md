# Contributers
* Mohamed El-Melegy
* Omar Khaled Mahmoud Safwat

# Executive Summary 
In this project we build a neural machine translation model with an RNN architecture, using LSTM. The project is built with Keras, a TensorFlow API, on a dataset that provides translation from the English language to Spanish.
# Motivation
Seq2Seq models utilizing LSTMs are considered state-of-art in Machine Translation. In this project, our aim is to gain a hands-on experience in building such model from scratch and test its performance, as we believe this will contribute heavily in building our intuition towards the structure and training of Neural Machine Translation models and open doors to understanding more sophisticated architectures utilizing attention models.


We then compare our custom model to a pretrained Transformer model, from [HuggingFace](https://huggingface.co/Helsinki-NLP/opus-mt-en-es).
# Procedure
## Data Collection
Data is provided from Anki, an app that helps users learn languages by creating their own flashcards with translations for different languages. The data is available at the following [link](https://www.manythings.org/anki/).
## Data Preprocessing
Data cleaning consists of the following:
1. Converting to Lowercase
2. Removing numbers and Punctuations
3. Removing extra spaces and special characters

Data Perprocessing:
1. Add the `START_` and the `_END` tags to the target sentences
1. Shuffle the data for better generalization
2. Split the Dataset into train and test
3. Pad sentances
4. Create Word Embedding; (256 dimensional vector)

## Model Training
Because the dataset is very large in size (> 132,000 lines), it will be difficult to train it in memory as one batch, this is why we use the `fit_generator()` method in keras and create a function to generate training batches.


As with any Sequence translation model, ours consists of an encoder LSTM cells followed by decoder for translation. 


### Encoder
Takes an input 2D array with a shape of (batch_size, max source sentence length).

### Decoder
Takes an input 2D array with a shape of (batch_size, max target sentence length).

Gives an outputs 3D array with a shape (batch_size, max target sentence length, number of unique words in target sentences)

We also use the **Teacher forcing** algorithm, which trains each cell in the decoder by supplying the actual value of the previous timestamp as input rather than the predicted output from the last time step.


Every LSTM cell in the decoder network outputs a dense softmax vector where most probable words are assigned higher probabilities.

### Optimization
The model is optimized using an RMSprop gradient descent instead of Adam to decrease computational cost.

# Conclusion and Results
Training a sequence to sequence Neural Translation model is indeed a very expensive operation; when considering the entirety of the 130,000 lines dataset, it was estimated that training would take more than 10 hours on a Google Colab GPU free instance, this is why we had to reduce the number of training samples as well as settle for a total of 128 batch size and 100 Epoch.


To enhance model performance we recommend doing the following:
1. Train on a larger portion of the dataset.
2. Train on more Epochs.
3. Consider using Adam optimizer.
4. Decrease Batch size.
5. Increase number of Neurons in LSTM cells.

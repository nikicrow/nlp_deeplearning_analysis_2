## Natural Language Processing using Deep Learning Algorithms for Emotion Recognition from text.

This model was developed in order to take any array of feedback or text in general and then generate what percentage of it had one of 5 emotions labelled against it.

The emotions are namely:
1. Anger
2. Fear
3. Joy
4. Neutral
5. Sadness

I have then created a spider chart to take these emotions and the percentages and then visualise the output.

# Process
1. Clean the text - use stem words and stopwords and remove special characters
2. Create Tokenizer using only the training data and tokenize
3. Define Model parameters
4. Create model - I used LSTM CNN but I should look into others too
5. Tokenize and pad X_train and X_test
6. Encode y_train and y_test
7. Create the model with an ideal number of epochs such that the validation loss isnt increasing (that means it may be overfit)
8. Feed the predicted emotions into a fancy spider chart maker
9. Visualise and enjoy!

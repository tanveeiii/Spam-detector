# SPAM-DETECTOR WEBSITE
### This website is used to detect whether an email is a spam or not. It uses machine learning algorithms from scikit-learn to train the models on the dataset.
### The training and test sets can be found in the [train-mails](train-mails) and [test-mails](test-mails) folders respectively.
### These folders contains multiple text files. Firstly, the content of all the text files are combined to a single pandas dataframe along with the label encoding 1 for a spam email and 0 for a ham email. The text column is then vectorized using TfidfVectorizer and then the models are trained on this data.
### The highest acccuracy of 0.9846 was obtained by using the SVM classifer.
### The home page consists of a text area where the user inputs the email he/she wants to check. Upon submitting, the result whether it is a spam or a ham email is displayed along with a feedback option. The feedback feature is used to improve the model based on the user's feedback.
### For example, if the result displayed is spam, and the user clicks on the dislike button, that means it is actually a ham email. So, the email is first preprocessd and then the program creates a new file in the ham label and write the preprocessed email into it. The model is then retrained. 

## Tech Stacks:
### Machine Learning - Python
### Frontend - HTML, CSS, Javascript
### Backend - Flask

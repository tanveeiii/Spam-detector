from flask import Flask, render_template, request
import trainModel

app = Flask(__name__)

svc, tfidf_vectorizer = trainModel.train_data()

@app.route('/', methods=["GET", "POST"])
def home():
    if(request.method=="GET"):
      return render_template("home.html")
    elif(request.method=="POST"):
       email_content = request.form.get('email')
       print(tfidf_vectorizer)
       result = trainModel.training(email_content, svc, tfidf_vectorizer)
       return result


if __name__ == "__main__":
  app.run(debug=True)
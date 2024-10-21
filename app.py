from flask import Flask, render_template, request
import trainModel

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def home():
    if(request.method=="GET"):
      return render_template("home.html")
    elif(request.method=="POST"):
       email_content = request.form.get('email')
       result = trainModel.training(email_content)
       return result


if __name__ == "__main__":
  app.run(debug=True)
from flask import Flask, render_template, request, redirect
import uuid
import trainModel

app = Flask(__name__)

svc, tfidf_vectorizer = trainModel.train_data()


@app.route('/', methods=["GET", "POST"])
def home():
    if(request.method=="GET"):
      return render_template("home.html")
    elif(request.method=="POST"):
       email_content = request.form.get('email')
       result = trainModel.training(email_content, svc, tfidf_vectorizer)
      #  print(result)
       return render_template("result.html", email_type= result, email=email_content)

@app.route('/feedback', methods=["POST"])
def feedback():
   if(request.method=="POST"):
      feedback = request.form.get("feedback")
      email = request.form.get("email")
      result = request.form.get("result_type")
      unique_id = str(uuid.uuid4())
      email = trainModel.preprocess_email_content(email)
      if(result=="spam"):
        if(feedback=="like"):
          with open(f"train-mails/spmsga{unique_id}.txt", "w", encoding="utf-8") as file:
            file.write(email)
            print("success1")
        else:
           with open(f"train-mails/9-{unique_id}msg.txt","w", encoding="utf-8") as file:
              file.write(email)
              print("success2")
      else:
        if(feedback=="like"):
          with open(f"train-mails/9-{unique_id}msg.txt","w", encoding="utf-8") as file:
            file.write(email)
            print("success3")
        else:
           with open(f"train-mails/spmsga{unique_id}.txt", "w", encoding="utf-8") as file:
            file.write(email)
            print("success4")
      
      trainModel.train_data()
      message="Thanks for sharing your feedback"
      return render_template("result.html", email_type= result, email=email, message=message)

if __name__ == "__main__":
  app.run(debug=True)
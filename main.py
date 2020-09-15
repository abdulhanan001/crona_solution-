from flask import Flask, render_template,request

app = Flask(__name__)
import pickle


#open file where you want to store data
file = open('model.pkl','rb')

clf = pickle.load(file)


file.close()

@app.route('/',methods= ["GET","POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])

        #code for inference
        inputFeature = [fever, age,pain , runnyNose, diffBreath]
        infProb = clf.predict_proba([inputFeature])[0][1]  # giving  input value to modle
        return render_template('show.html', inf=(infProb))

    return render_template('index.html')


#@app.route('/',methods= ["GET","POST"])

    #return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)

print('hello hann')
from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import joblib
model = pickle.load(open("mpg.pkl",'rb'))
#model = pickle.load(open("profit.pkl",'rb'))
ct = joblib.load('column1')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/guest', methods =["post"])
def Guest():
    eng = request.form["eng"]
    hp = request.form["hp"]
    vol = request.form["vol"]
    sp = request.form["sp"]
    wt = request.form["wt"]
    data =[[eng,hp,vol,sp,wt]]
    prediction = model.predict(ct.transform(data))
    prediction = prediction[0]
    return render_template("index.html",y = "milage could be" + ' ' + str(prediction))
if __name__ == '__main__':

    app.run(debug = True)

                           #@app.route('/user')
#def user ():
   # return "hellow user welcome"

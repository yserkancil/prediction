from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Modelin eğitimini burada yapabilirsiniz. 
# Bu örnekte model her seferinde eğitiliyor. Gerçek bir projede modeli eğitip kaydetmek ve yüklemek daha verimli olur.
def heart_attack_prediction(oldpeak, thal, cp, thalach, ca):
    # Model ve veri seti
    veriler = pd.read_csv('heart.csv')  # heart.csv dosyanızın mevcut olduğundan emin olun
    X = veriler[['oldpeak', 'thal', 'cp', 'thalach', 'ca']]
    y = veriler['target']
    model = RandomForestClassifier(random_state=0)
    model.fit(X, y)
    prediction = model.predict([[oldpeak, thal, cp, thalach, ca]])
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kalpkrizi', methods=['GET', 'POST'])
def kalpkrizi():
    if request.method == 'POST':
        oldpeak = float(request.form['textbox1'])
        thal = float(request.form['textbox2'])
        cp = float(request.form['textbox3'])
        thalach = float(request.form['textbox4'])
        ca = float(request.form['textbox5'])
        prediction = heart_attack_prediction(oldpeak, thal, cp, thalach, ca)
        result_text = "Kalp krizi riski var" if prediction == 1 else "Kalp krizi riski yok"
        return render_template('result.html', result=result_text)
    return render_template('kalpkrizi.html')

if __name__ == '__main__':
    app.run(debug=True)




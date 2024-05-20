import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Kalp krizi riski tahmini fonksiyonu
def heart_attack_prediction(oldpeak, thal, cp, thalach, ca):
    # Eğitilmiş model ve gerekli veri setini yükle
    veriler = pd.read_csv('heart.csv')
    veriler = veriler.select_dtypes(include=['float64', 'int64'])
    veriler = null_kontrol_doldur(veriler)

    # Özelliklerin seçilmesi ve veri setinin oluşturulması
    selected_features = veriler[['oldpeak', 'thal', 'cp', 'thalach', 'ca']]
    dependent_variables = veriler['target']

    # Modeli eğit
    model = RandomForestClassifier(random_state=0)
    model.fit(selected_features, dependent_variables)

    # Kullanıcının girdiği değerlerle bir veri çerçevesi oluştur
    kullanici_verisi = pd.DataFrame({'oldpeak': [oldpeak], 'thal': [thal], 'cp': [cp], 'thalach': [thalach], 'ca': [ca]})

    # Tahmin yap
    tahmin = model.predict(kullanici_verisi)

    return tahmin[0]

# Veri setindeki null değerleri kontrol eden ve gerekli olanları sütun ortalaması ile dolduran fonksiyon
def null_kontrol_doldur(veri_seti):
    null_degerler = veri_seti.isnull().sum()
    for sutun in veri_seti.columns:
        if null_degerler[sutun] > 0:
            sutun_ort = veri_seti[sutun].mean()
            veri_seti[sutun].fillna(sutun_ort, inplace=True)
    return veri_seti

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kullanıcı girdilerini al
        oldpeak = float(request.form['oldpeak'])
        thal = float(request.form['thal'])
        cp = float(request.form['cp'])
        thalach = float(request.form['thalach'])
        ca = float(request.form['ca'])

        # Tahmini yap
        tahmin = heart_attack_prediction(oldpeak, thal, cp, thalach, ca)

        # Tahmin sonucunu göster
        if tahmin == 1:
            result = "Kalp krizi riski var."
        else:
            result = "Kalp krizi riski yok."
        
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

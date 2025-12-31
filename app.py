from flask import Flask,render_template,request
import pandas as pd
from src.utils import load_object

app=Flask(__name__)

model=load_object('model/cuisine_pipeline.pkl')

@app.route('/',methods=['GET','POST'])
def index():
    prediction=None

    if request.method == 'POST':
        data={
            'City':request.form['city'],
            'Average Cost for two':float(request.form['cost']),
            'Price range':float(request.form['price_range']),
            'Has Table booking':request.form['table'],
            'Has Online delivery':request.form['delivery'],
            'Aggregate rating':float(request.form['rating']),
            'Votes':int(request.form['votes'])
        }

        df = pd.DataFrame([data])

        probs = model.predict_proba(df)[0]
        classes = model.classes_

        top3_idx = probs.argsort()[-3:][::-1]

        prediction = [
            (classes[i], round(probs[i] * 100, 2))
            for i in top3_idx
        ]

    return render_template('home.html',predictions=prediction)

if __name__ == '__main__':
    app.run(debug=True)

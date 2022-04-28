from flask import Flask,render_template,request
import pickle

identity_model = pickle.load(open('identity.pkl','rb'))

insult_model = pickle.load(open('insult.pkl','rb'))

obsence_model = pickle.load(open('obsence.pkl','rb'))

severe_model = pickle.load(open('severe.pkl','rb'))

toxic_model = pickle.load(open('toxic.pkl','rb'))

threat_model = pickle.load(open('threat.pkl','rb'))

from identity_preprocess import identity_preprocessing
from insult_preprocess import insult_preprocessing
from severe_preprocess import severe_preprocessing
from obsence_preprocess import obsence_preprocessing 
from toxic_preprocess import toxic_preprocessing
from threat_preprocess import threat_preprocessing


app = Flask(__name__)






@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
        result = []
        message = request.form['textarea']
        identity_prediction = identity_model.predict(identity_preprocessing(message))
        insult_prediction = insult_model.predict(insult_preprocessing(message))
        obsence_prediction = obsence_model.predict(obsence_preprocessing(message))
        severe_prediction = severe_model.predict(severe_preprocessing(message))
        toxic_prediction = toxic_model.predict(toxic_preprocessing(message))
        threat_prediction = threat_model.predict(threat_preprocessing(message))
        print(identity_prediction)
        print(insult_prediction)
        print(obsence_prediction)
        print(severe_prediction)
        print(toxic_prediction)
        print(threat_prediction)

        if identity_prediction[0] == 0 :
            result.append(" Not Identity")
        else:
            result.append("Identity Hate")
        
        if insult_prediction[0] == 0 :
            result.append("Not Insult")
        else:
            result.append("Insult")
        
        if obsence_prediction[0] == 0 :
            result.append("Not Obsence")
        else:
            result.append("Obsence")
        
        if severe_prediction[0] == 0 :
            result.append("Not Severe")
        else:
            result.append("Severe")
        
        if toxic_prediction[0] == 0 :
            result.append("Not Toxic")
        else:
            result.append("Toxic")
        
        if threat_prediction[0] == 0 :
            result.append("Not Threat")
        else:
            result.append("Threat")
        
        print(result)
        return render_template('index.html',result=result)
        

   
    
           
    


if __name__ == '__main__':
    app.run(debug=False)

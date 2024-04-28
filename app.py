from imports import *
from utlis import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/sentiment')
def sentiment():
    return render_template("sentiment.html")

@app.route('/chatsentiment', methods=['GET','POST'])
def chatsentiment():
    ExportedFile=CustomData( ExportedFile = request.files['ExportedFile'])
    ExportedFile.SaveFile()

    preprocess=PreprocessPipeline()
    conversation,urls=preprocess.Preprocess()

    print(conversation)

    
    sentiment=SentimentPipeline(conversation)
    sentiment_predictions=sentiment.Predict()

    sentiment_result=""
    non_toxic=""
    toxic=""
    severe_toxic=""
    obscene=""
    threat=""
    insult=""
    identity_hate=""

    for i in sentiment_predictions:
        if i['label']=="toxic":
            toxic="Toxic : "+str(i['score'])
        elif i['label']=="severe_toxic":
            severe_toxic="Severe Toxic : "+str(i['score'])
        elif i['label']=="obscene":
            obscene="Obscene : "+str(i['score'])
        elif i['label']=="threat":
            threat="Threat : "+str(i['score'])
        elif i['label']=="insult":
            insult="Insult : "+str(i['score'])
        elif i['label']=="identity_hate":
            identity_hate="Identity Hate : "+str(i['score'])

    if sentiment_predictions[0]['score']<0.5:
        sentiment_result="non-toxic"
        non_toxic="Non-Toxic"
    else :
        sentiment_result="toxic"
        
    phising=PhisingPipeline(urls)
    phising_predictions=phising.Predict()
    
    if phising_predictions[0]>0.5:
        phising_result="Websites are Benign"
    else:
        phising_result="Websites are Malware"

    Summarization=SummarizationPipeline(conversation)
    Summarization_result=Summarization.Predict()

    return render_template('sentiment.html',phising_result=phising_result,sentiment_result=sentiment_result,non_toxic=non_toxic,toxic=toxic,severe_toxic=severe_toxic,obscene=obscene,threat=threat,insult=insult,identity_hate=identity_hate,Summarization_result=Summarization_result)

if __name__ == '__main__':
    app.run(debug=True)

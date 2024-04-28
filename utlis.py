from imports import *

class PreprocessPipeline():
    def __init__(self):
        self.ExportedFilePath = os.path.join("artifacts", "ExportedFile.txt")
        self.raw_conversation = []
        self.conversation = ""
        self.URls=[]

    def ConvertEmojis(self):
        self.raw_conversation = emoji.demojize(self.raw_conversation)

    def RemoveURLs(self):
        url_pattern = r'https?://\S+|www\.\S+'
        self.URls = re.findall(url_pattern, self.raw_conversation)

    def RemoveStopWords(self):
        stop_words = set(stopwords.words('english'))
        self.raw_conversation = word_tokenize(self.raw_conversation)
        self.raw_conversation = [token.lower() for token in self.raw_conversation if token not in stop_words]

    def RemovePunctuationNumber(self):
        self.raw_conversation = [re.sub(r'[^\w\s]', '', token) for token in self.raw_conversation]
        self.raw_conversation = [re.sub(r'[\d]', '', token) for token in self.raw_conversation]
        
    def Lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        self.raw_conversation = [lemmatizer.lemmatize(token) for token in self.raw_conversation]
        self.conversation = " ".join(self.raw_conversation)

    def Preprocess(self):
        rows=[]
        current_date = None
        current_time = None
        current_username = None
        current_msg = None
        pattern = r'(\d{1,2}/\d{1,2}/\d{2}),?\s(\d{1,2}:\d{2}\s?[APMapm]{2})\s-\s(.+?):\s(.+)'
        file = open(self.ExportedFilePath, 'r', encoding='utf-8')
        data = file.readlines()
        file.close()
        for line in data:
            match = re.search(pattern, line)
            if match is not None:
                if current_date is not None:
                    rows.append({
                        'only_date': current_date,
                        'time': current_time,
                        'username': current_username,
                        'message': current_msg
                    })
                date = match.group(1)
                time = match.group(2)
                username = match.group(3)
                msg = match.group(4)
                current_date = date
                current_time = time
                current_username = username
                current_msg = msg
            else:
                if current_msg is not None:
                    current_msg += ' ' + line.strip()

        if current_date is not None:
            rows.append({
                'only_date': current_date,
                'time': current_time,
                'username': current_username,
                'message': current_msg
            })
        df = pd.DataFrame(rows)
        for i in df['message']:
            self.raw_conversation+=i+" "
        self.ConvertEmojis()
        self.RemoveURLs()
        self.RemoveStopWords()
        self.RemovePunctuationNumber()
        self.Lemmatize()
        return self.conversation,self.URls

class SentimentPipeline():
    def __init__(self, conversation):
        self.conversation = conversation
        self.sentiment = ""
        self.tokenizer = AutoTokenizer.from_pretrained("models/Sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("models/Sentiment")
        self.labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def calculate_mean(self, predictions):
        num_sentences = len(predictions)
        labels = [label_prediction['label'] for label_prediction in predictions[0]]
        sum_scores = {label: 0.0 for label in labels}
        for sentence_predictions in predictions:
            for label_prediction in sentence_predictions:
                label = label_prediction['label']
                score = label_prediction['score']
                sum_scores[label] += score
        mean_predictions = [{'label': label, 'score': sum_scores[label] / num_sentences} for label in labels]
        return mean_predictions

    def Predict(self):
        # split the data into chunks of 512 len to avoid the max length of the model
        sentences = [self.conversation[i:i+512] for i in range(0, len(self.conversation), 512)]
        predictions = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probabilities = torch.sigmoid(logits)
            prediction = [{"label": label, "score": probabilities[0][i].item()} for i, label in enumerate(self.labels)]
            predictions.append(prediction)
        mean_predictions = self.calculate_mean(predictions)
        return mean_predictions
    
class PhisingPipeline():
    def __init__(self,urls):
        self.urls = urls
        self.tokenizer = AutoTokenizer.from_pretrained("models/Phising")
        self.model = AutoModelForSequenceClassification.from_pretrained("models/Phising")
        self.lables=["benign", "malware"]

    def Predict(self):
        benign_predictions = []
        malware_predictions = []
        for i in self.urls:
            inputs = self.tokenizer(i, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
            prediction = torch.softmax(logits, dim=1).tolist()[0]
            benign_predictions.append(prediction[0])
            malware_predictions.append(prediction[1])
        return [mean(benign_predictions), mean(malware_predictions)]

class SummarizationPipeline():
    def __init__(self,conversation):
        self.conversation = conversation
        self.summarization = ""
        self.tokenizer = AutoTokenizer.from_pretrained("models/Summarization")
        self.model = BartForConditionalGeneration.from_pretrained("models/Summarization")

    def Predict(self):
        input_ids = self.tokenizer.encode(self.conversation, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(input_ids,max_length=150,min_length=40,length_penalty=2.0,num_beams=4,early_stopping=True)
        self.summarization = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return self.summarization

class CustomData:
    def __init__(self, ExportedFile):
        self.ExportedFile = ExportedFile

    def SaveFile(self):
        self.ExportedFile.save(os.path.join("artifacts", "ExportedFile.txt"))
        return "File Saved"
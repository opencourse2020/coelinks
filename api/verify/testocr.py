from models import Document
from api.verify.views import preprocess_image
import keras_ocr
import pandas as pd
import os
import glob



docs = Document.objects.filter(verified=True, scanned=False)
# doc = Document.objects.create(user=19, keyword='Test 22')

for doc in docs:
    userid = doc.user
    kw = doc.keyword
    print("user Id:")
    print(userid)
    print(kw)
    path = os.getcwd() + "/media/documents/user_" + str(userid) + "/*"
    status = ""
    pipeline = keras_ocr.pipeline.Pipeline()
    for path_to_document in glob.glob(path, recursive=True):
        images = [keras_ocr.tools.read(img) for img in [path_to_document]]
        prediction_groups = pipeline.recognize(images)
        df = pd.DataFrame(prediction_groups[0], columns=['text', 'bbox'])
        nameexist = kw in df['text'].values
        if nameexist:
            status = "Verified"
            with open('verifieds.txt', 'a') as f:
                f.writelines(docs.user)
    print(status)
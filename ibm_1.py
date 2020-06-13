
import json
from ibm_watson import VisualRecognitionV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator('QeULZ3tNyXhuGlh4eQWn0DTJgcQET86_fym6s3Yn_A8z')
visual_recognition = VisualRecognitionV3(
    version='2018-03-19',
    authenticator=authenticator
)

visual_recognition.set_service_url('https://api.us-south.visual-recognition.watson.cloud.ibm.com/instances/e83c2dab-5735-4074-8358-f792e6c3dc47')


with open('/Users/sushmabansal/Desktop/food_data/images/apple_pie/134.jpg', 'rb') as images_file:
    classifier_ids = ["food"]
    classes_result = visual_recognition.classify(images_file=images_file, classifier_ids=classifier_ids).get_result()
    print(json.dumps(classes_result, indent=2))
   

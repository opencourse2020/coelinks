from django.shortcuts import render
import os, shutil
from django.conf import settings
from django.http import JsonResponse
# Create your views here.
from rest_framework import authentication, permissions
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer, FileScanSerializer, HeadShotSerializer
from django.views.decorators.csrf import csrf_exempt
from .models import Document, AIModel, HeadShot
import re
import os.path as osp
import difflib
from io import StringIO
import hmac
import hashlib
from base64 import b64encode
# AI libraries
import glob
import pandas as pd
import cv2
import numpy as np
import pytesseract
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from . import RRDBNet_arch as arch
# from piq import niqe
# import easyocr
# import keras_ocr
# from numpy.lib.polynomial import poly
# import matplotlib.pyplot as plt
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# from ultralytics import YOLO
# from ultralytics import settings as sts
from super_gradients.training import models


classes = [
          'Birth Certificate',
          'ID/DL',
          'Invalid']

static_folder = settings.STATIC_ROOT
media_folder = settings.MEDIA_ROOT

picture_id_model = static_folder + "/models/best_model.pth"
picture_enhance_model = "models/RRDB_ESRGAN_x4.pth"

ai_model = torch.load(picture_id_model)
yolov8_model: str = os.path.join(settings.BASE_DIR, 'media', 'aimodels/')
yolov8_run: str = os.path.join(settings.BASE_DIR, 'media', 'aimodels')


mean = [0.7683, 0.7671, 0.7645]
std = [0.2018, 0.1925, 0.1875]

image_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


def verifySignature(receivedSignature: str, secret, params):

  data = "-".join(params)
  data = data.encode('utf-8')
  computed_sig = hmac.new(secret.encode('utf-8'), msg=data, digestmod=hashlib.sha256).digest()
  signature = b64encode(computed_sig).decode()
  if signature == receivedSignature:
    return True
  return False



def set_device():
  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  return torch.device(dev)


def is_head_shot_clear(image_path, threshold=20):
  # path = os.getcwd() + "/media/headshots/user_" + str(userid) + "/*"
  # image = ""
  # for image_path in glob.glob(path, recursive=True):
    # Load the image using OpenCV
  image = cv2.imread(image_path)

  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Calculate the Variance of Laplacian to measure image clarity
  variance_of_laplacian = cv2.Laplacian(gray_image, cv2.CV_64F).var()

  # Determine if the image is clear based on the threshold
  is_clear = variance_of_laplacian > threshold
  # is_clear = variance_of_laplacian

  return is_clear


def headshots_count(image_path):
  # path = os.getcwd() + "/media/headshots/user_" + str(userid) + "/*"
  # image = ""
  # for image_path in glob.glob(path, recursive=True):
    # Load the image using OpenCV
  #image = cv2.imread(image_path)

  # Convert the image to grayscale
  # image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Method 1
  # Call Yolo V4 to detect objects in the image
  # print("Start image detection")
  # boxes, label, count = cv.detect_common_objects(image)
  # print("Number of boxes")
  # print(len(boxes))
  ## output = draw_bbox(image, box, label, count)

# Method 2
  # Call Yolo V8 to detect objects in the image
  # sts.update({'runs_dir': yolov8_run})
  # sts.reset()
  # model = YOLO(yolov8_model + "yolov8s.pt")
  # results = model.predict(source=image_path, conf=0.3)
  # boxes = results[0].boxes

# Method 3
  # call Yolo Nas to detect objects in the image
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  MODEL_ARCH = 'yolo_nas_l'
  model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)
  CONFIDENCE_TRESHOLD = 0.35
  result = list(model.predict(image_path, conf=CONFIDENCE_TRESHOLD))[0]
  dp = result.prediction
  boxes = dp.bboxes_xyxy
  # Determine if the image is clear based on the threshold
  one_person = len(boxes) == 1

  return one_person


def is_image_clear(image_path, threshold=3.5):
  # Load the image using Pillow (PIL)
  image = Image.open(image_path).convert('RGB')

  # Convert the image to a PyTorch tensor
  image_tensor = torch.tensor([transforms.ToTensor()(image)])

  # Load the pre-trained NIQE model
  niqe_model = niqe()

  # Calculate the NIQE score for the image
  niqe_score = niqe_model(image_tensor).item()

  # Determine if the image is clear based on the threshold
  is_clear = niqe_score < threshold

  return is_clear


def enhancepictures(userid):
  model_path = picture_enhance_model  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
  device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
  # device = torch.device('cpu')

  print("current folder")
  print(model_path)
  test_img_folder = media_folder + "/documents/user_" + str(userid) + "/*"
  test_user_folder = media_folder + "/documents/user_" + str(userid) + "/"
  print(test_user_folder)
  model = arch.RRDBNet(3, 3, 64, 23, gc=32)
  model.load_state_dict(torch.load(model_path), strict=True)
  model.eval()
  model = model.to(device)

  print('Model path {:s}. \nTesting...'.format(model_path))

  idx = 0
  im_path = ""
  for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    print("Finished")
    with torch.no_grad():
      output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    print("Write picture to file")
    cv2.imwrite('{pth}{file}_rlt.png'.format(pth=test_user_folder, file=base), output)
    im_path = '{pth}{file}_rlt.png'.format(pth=test_user_folder, file=base)
  return im_path

def classify(aimodel, image_transforms, image_path, classes):
  aimodel = aimodel.eval()
  image = Image.open(image_path)
  image = image_transforms(image).float()
  image = image.unsqueeze(0)

  output = aimodel(image)
  _, predicted = torch.max(output.data, 1)

  return classes[predicted.item()]


class FileUpdateView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  # parser_classes = (FileUploadParser,)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)

    # Check if call is authorized
    # *******************************************************************************************************
    api_signature = request.headers['Authorization']
    if (api_signature is None) or (api_signature == ""):
      return Response({"Fail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    sha_name, signature = api_signature.split("=", 1)
    if sha_name != "sha256":
      return Response({"Fail": "Operation not supported."}, status=status.HTTP_501_NOT_IMPLEMENTED)

    secret = settings.UPLOADDOCUMENTKEY
    params = [secret, request.method, request.path]
    is_valid = verifySignature(signature, secret, params)
    if is_valid != True:
      return Response({"Fail": "Invalid signature. Permission denied."}, status=status.HTTP_403_FORBIDDEN)
    # *******************************************************************************************************

    if file_serializer.is_valid():
      userid = int(file_serializer.data['user'])
      # key_word = file_serializer.data['keyword']
      filename = file_serializer.validated_data['file']
      # print("__name__")
      # print(__name__)
      verified = classify(ai_model, image_transforms, filename, classes)
      # if verified == "Invalid":
      doc = Document.objects.filter(user=userid).first()
      if doc:
        delete(userid, 1)
        if verified == "Invalid":
          obj, created = Document.objects.update_or_create(
            user=userid,
            defaults={'verified': False, 'file': filename, 'scanned': False},
          )
          verified = verified + " - https://coelink.com" + obj.file.url
        elif verified != "Invalid":

          obj, created = Document.objects.update_or_create(
            user=userid,
            defaults={'verified': True, 'file': filename, 'scanned': False},
          )
          checkTextinImage = Checkpicture(userid)
          verified = "Valid"
          if not checkTextinImage:
            obj, created = Document.objects.update_or_create(
              user=userid,
              defaults={'verified': False},
            )
            verified = "Invalid"
          verified = verified + " - https://coelink.com" + obj.file.url
      elif doc is None and verified == "Invalid":
        obj, created = Document.objects.update_or_create(
          user=userid,
          defaults={'verified': False, 'file': filename},
        )
        verified = verified + " - https://coelink.com" + obj.file.url
      elif doc is None and verified != "Invalid":
        obj, created = Document.objects.update_or_create(
          user=userid,
          defaults={'verified': True, 'file': filename},
        )
        checkTextinImage = Checkpicture(userid)
        verified = "Valid"
        if not checkTextinImage:
          obj, created = Document.objects.update_or_create(
            user=userid,
            defaults={'verified': False},
          )
          verified = "Invalid"
        verified = verified + " - https://coelink.com" + obj.file.url

      # print("step 1")
      # enhancepictures(userid)
      # print("step 2")
      return Response(verified, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# Create your views here.


class DocumentScanView(APIView):
  parser_classes = (MultiPartParser, FormParser)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileScanSerializer(data=request.data)

    # Check if call is authorized
    # *******************************************************************************************************
    api_signature = request.headers['Authorization']
    if (api_signature is None) or (api_signature == ""):
      return Response({"Fail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    sha_name, signature = api_signature.split("=", 1)
    if sha_name != "sha256":
      return Response({"Fail": "Operation not supported."}, status=status.HTTP_501_NOT_IMPLEMENTED)

    secret = settings.SCANDOCUMENTKEY
    params = [secret, request.method, request.path]
    is_valid = verifySignature(signature, secret, params)
    if is_valid != True:
      return Response({"Fail": "Invalid signature. Permission denied."}, status=status.HTTP_403_FORBIDDEN)
    # *******************************************************************************************************

    if file_serializer.is_valid():
      userid = int(file_serializer.data['user'])
      key_word = file_serializer.data['keyword']

      scanned = "Verified"
      doc = Document.objects.filter(user=userid).first()

      if doc and doc.verified == True:
        scanned = Scanpicture(key_word, userid)

        scanned = scanned + doc.file.url
        # obj, created = Document.objects.update_or_create(
        #   user=userid,
        #   defaults={'scanned': True},
        # )
        doc.scanned = True
        doc.save()
      elif doc and doc.verified == False:
        return Response({"Fail": "Document not verified yet."}, status=status.HTTP_403_FORBIDDEN)
      elif doc is None:
        return Response({"Fail": "No File to Scan"}, status=status.HTTP_400_BAD_REQUEST)

      return Response(scanned, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PictureVerifyView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  # parser_classes = (FileUploadParser,)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = HeadShotSerializer(data=request.data)

    # Check if call is authorized
    # *******************************************************************************************************
    api_signature = request.headers['Authorization']
    if (api_signature is None) or (api_signature == ""):
      return Response({"Fail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    sha_name, signature = api_signature.split("=", 1)
    if sha_name != "sha256":
      return Response({"Fail": "Operation not supported."}, status=status.HTTP_501_NOT_IMPLEMENTED)

    secret = settings.PICTUREVERIFYKEY
    params = [secret, request.method, request.path]
    is_valid = verifySignature(signature, secret, params)
    if is_valid != True:
      return Response({"Fail": "Invalid signature. Permission denied."}, status=status.HTTP_403_FORBIDDEN)
    # *******************************************************************************************************


    if file_serializer.is_valid():
      userid = int(file_serializer.data['user'])
      filename = file_serializer.validated_data['file']
      # key_word = file_serializer.data['keyword']
      doc = HeadShot.objects.filter(user=userid).first()
      if doc:
        delete(userid, 2)

      obj, created = HeadShot.objects.update_or_create(
        user=userid,
        defaults={'file': filename},
      )
      # is_clear = True

      verified = ""
      is_clear = is_head_shot_clear(obj.file.path)
      one_person = headshots_count(obj.file.path)
      # is_clear = False
      if is_clear and one_person:
        verified = "1 - https://coelink.com" + obj.file.url
        # obj, created = HeadShot.objects.update_or_create(
        #   user=userid,
        #   defaults={'verified': True},
        # )
        obj.delete()
        delete(userid, 2)
      elif not one_person:
        verified = "2 - https://coelink.com" + obj.file.url
        # obj, created = HeadShot.objects.update_or_create(
        #   user=userid,
        #   defaults={'verified': False},
        # )
      elif not is_clear:
        verified = "3 - https://coelink.com" + obj.file.url
        # verified = is_clear
        # obj, created = HeadShot.objects.update_or_create(
        #   user=userid,
        #   defaults={'verified': False},
        # )
      return Response(verified, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class DocumentVerifiedView(APIView):
  parser_classes = (MultiPartParser, FormParser)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileScanSerializer(data=request.data)

    # Check if call is authorized
    # *******************************************************************************************************
    api_signature = request.headers['Authorization']
    if (api_signature is None) or (api_signature == ""):
      return Response({"Fail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    sha_name, signature = api_signature.split("=", 1)
    if sha_name != "sha256":
      return Response({"Fail": "Operation not supported."}, status=status.HTTP_501_NOT_IMPLEMENTED)

    secret = settings.DELETEDOCUMENTKEY
    params = [secret, request.method, request.path]
    is_valid = verifySignature(signature, secret, params)
    if is_valid != True:
      return Response({"Fail": "Invalid signature. Permission denied."}, status=status.HTTP_403_FORBIDDEN)
    # *******************************************************************************************************

    if file_serializer.is_valid():
      userid = int(file_serializer.data['user'])

      doc = Document.objects.filter(user=userid).first()
      deleted = "Notfound"

      if doc:
        doc.delete()
        delete(userid, 1)

        deleted = "Done"



        return Response(deleted, status=status.HTTP_200_OK)
      else:
        return Response(deleted, status=status.HTTP_404_NOT_FOUND)

    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class HeadshotVerifiedView(APIView):
  parser_classes = (MultiPartParser, FormParser)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileScanSerializer(data=request.data)

    # Check if call is authorized
    # *******************************************************************************************************
    api_signature = request.headers['Authorization']
    if (api_signature is None) or (api_signature == ""):
      return Response({"Fail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    sha_name, signature = api_signature.split("=", 1)
    if sha_name != "sha256":
      return Response({"Fail": "Operation not supported."}, status=status.HTTP_501_NOT_IMPLEMENTED)

    secret = settings.DELETEDOCUMENTKEY
    params = [secret, request.method, '/api/endverify/']
    is_valid = verifySignature(signature, secret, params)
    if is_valid != True:
      return Response({"Fail": "Invalid signature. Permission denied."}, status=status.HTTP_403_FORBIDDEN)
    # *******************************************************************************************************

    if file_serializer.is_valid():
      userid = file_serializer.data['user']

      doc = HeadShot.objects.filter(user=userid).first()
      deleted = ""
      if doc:
        doc.delete()
        delete(userid, 2)

        deleted = "Done"

      return Response(deleted, status=status.HTTP_200_OK)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def find_string(text, target_string):
  # Convert both the text and target string to lowercase for case-insensitive matching
  text_lower = text.lower()
  target_string_lower = target_string.lower()

  # Search for the target string within the text
  if target_string_lower in text_lower:
    return True
  else:
    return False


def Scanpicture(athname, userid):
  # athname = request.POST.get('athname')
  # path = os.getcwd() + "/media/documents/*"
  # test_user_folder = media_folder + "/documents/user_" + str(userid) + "/"
  # folder = os.getcwd() + '/media/documents/user_' + str(userid) + '/'
  # for filename in os.listdir(folder):
  #   path_to_document = folder + filename
  path = os.getcwd() + "/media/documents/user_" + str(userid) + "/*"
  filter_predicted_result = ""
  for path_to_document in glob.glob(path, recursive=True):
  #   # img = cv2.imread(path_to_document)
    img = preprocess_image(path_to_document)

    # pytesseract method
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    predicted_result = pytesseract.image_to_string(img, lang='eng')

    # Keras OCR method
    # print("path to document")
    # print(path_to_document)
    # pipeline = keras_ocr.pipeline.Pipeline()
    # images = [keras_ocr.tools.read(img) for img in [path_to_document]]
    # prediction_groups = pipeline.recognize(images)
    # print("Finished")
    # df = pd.DataFrame(prediction_groups[0], columns=['text', 'bbox'])
    # print(df)
    # reader = easyocr.Reader(['en'])
    # predicted_result = reader.readtext(img)


    # predicted_result = pytesseract.image_to_string(img, lang='eng',config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    filter_predicted_result = "".join(predicted_result.split("\n")).replace(":", "")\
      .replace("-", "").replace("”", "").replace("“", "").replace(">", "").replace(")", "").replace("(", "")

  words = athname.split()

  status = ""

  for wd in words:
    nameexist = find_string(filter_predicted_result, wd)
    # nameexist = wd in df['text'].values
    if nameexist:
      # status = status + wd + " Verified - "
      status = "Verified - https://coelink.com"
    else:
      # Check if
      datax = list(map(lambda x: x.split(' '), filter_predicted_result.split("\r\n")))
      df = pd.DataFrame(datax[0])
      df[0] = df[0].map(str.lower)
      lwd= wd.lower()
      similar = difflib.get_close_matches(lwd, df[0].values)
      # similar = []
      if len(similar) > 0:
        # status = status + wd + " Verified - "
        status = "Verified - https://coelink.com"
      else:
        # status = status + wd + " Unverified - "
        status = "Unverified - https://coelink.com"
        # status = filter_predicted_result
        return status


  # context = {'filter_predicted_result': filter_predicted_result, 'name': name}
  # status = filter_predicted_result
  return status
  # context = {'form': form}
  # return render(request, 'homepage.html', context)


def Checkpicture(userid):

  path = os.getcwd() + "/media/documents/user_" + str(userid) + "/*"
  status = False
  for path_to_document in glob.glob(path, recursive=True):
    img = preprocess_image(path_to_document)

    # pytesseract method
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    predicted_result = pytesseract.image_to_string(img, lang='eng')

    filter_predicted_result = "".join(predicted_result.split("\n")).replace(":", "") \
      .replace("-", "").replace("”", "").replace("“", "").replace(">", "").replace(")", "").replace("(", "")

    if len(filter_predicted_result) > 0:
      status = True

  return status


def delete(userid, type):
  folder = ""
  if type == 1:
    folder = os.getcwd() + '/media/documents/user_' + str(userid) + '/'
  elif type == 2:
    folder = os.getcwd() + '/media/headshots/user_' + str(userid) + '/'
  if os.path.exists(folder):
    for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
          os.unlink(file_path)
        elif os.path.isdir(file_path):
          shutil.rmtree(file_path)
      except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))


def deletefile(file_path):
  try:
    if os.path.isfile(file_path) or os.path.islink(file_path):
      os.unlink(file_path)
    elif os.path.isdir(file_path):
      shutil.rmtree(file_path)
  except Exception as e:
    print('Failed to delete %s. Reason: %s' % (file_path, e))


def preprocess_image(image_path):
  # Load the image
  image = cv2.imread(image_path)

  # Resize the image for better OCR accuracy
  resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

  # Convert the image to grayscale
  gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

  # Denoise the image using a bilateral filter
  denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

  # Apply adaptive thresholding to enhance contrast
  threshold_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

  return threshold_image

# ********************************************************************************************
# ********************************************************************************************
# ******************************* Views to Handle Cron Jobs ****************************
# ********************************************************************************************
# ********************************************************************************************


class UnverifiedViewset(viewsets.ModelViewSet):
  serializer_class = FileSerializer

  def _get_unverified_users(self):
    docs = Document.objects.filter(verified=True)
    print(docs)
    return docs



# ********************************************************************************************
# *************************   Testing Apis   *************************************************

class FileUpdatetestView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  # parser_classes = (FileUploadParser,)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)

    # Check if call is authorized
    # *******************************************************************************************************
    api_signature = request.headers['Authorization']
    if (api_signature is None) or (api_signature == ""):
      return Response({"Fail": "Permission denied."}, status=status.HTTP_403_FORBIDDEN)

    sha_name, signature = api_signature.split("=", 1)
    if sha_name != "sha256":
      return Response({"Fail": "Operation not supported."}, status=status.HTTP_501_NOT_IMPLEMENTED)

    secret = settings.UPLOADDOCUMENTKEY
    params = [secret, request.method, request.path]
    is_valid = verifySignature(signature, secret, params)
    if is_valid != True:
      return Response({"Fail": "Invalid signature. Permission denied."}, status=status.HTTP_403_FORBIDDEN)
    # *******************************************************************************************************



    if file_serializer.is_valid():
      userid = file_serializer.data['user']
      key_word = file_serializer.data['keyword']
      filename = file_serializer.validated_data['file']

      # Step 1: Check if document is valid
      verified = classify(ai_model, image_transforms, filename, classes)

      # Step 2: Save the document to database if it's valid
      if verified != "Invalid":
        doc = Document.objects.filter(user=userid).first()
        if doc:
          delete(userid, 1)

        obj, created = Document.objects.update_or_create(
          user=userid,
          defaults={'keyword': key_word, 'file': filename},
        )
        # Step 3: Check if the document quality is good

        is_clear = is_head_shot_clear(obj.file.path)
        print("is clear: ")
        print(is_clear)
        # if not is_clear:
        # im_path = enhancepictures(userid)
        # print(im_path)
        # if im_path != "":
        #   old_file = obj.file.path
        #   deletefile(obj.file.path)

        #   os.rename(im_path, old_file)

      # Stp 5: Return result
      return Response(verified, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# Create your views here.


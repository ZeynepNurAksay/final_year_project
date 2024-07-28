# necessary imports
from django.shortcuts import render
from django import forms
from django.core.files.storage import FileSystemStorage
from  django.http import HttpResponse
import shutil
import os
import cv2
from ultralytics import YOLO
from PIL import Image
from pdf2image import convert_from_path


# django form for uploading files
class uploadFileForm(forms.Form):
    file = forms.FileField(widget= forms.FileInput(attrs={'id':'european_file'}))
    
# deletes all media directory and runs directory contents to clear harddisk.
def delete_all_previous_predictions_and_saved_images():
    isExist = os.path.exists('media')
    if isExist:
        shutil.rmtree('media')

    isExist2 = os.path.exists('runs')
    if isExist2:
        shutil.rmtree('runs')

# deep learning can be applied on images, so in this function we convert all the page
# in pdf to .jpg image file
def pdf_to_image(path):
    page_names = []
    pages = convert_from_path(path)
    for i in range(len(pages)):
        pages[i].save('media/page'+ str(i) +'.jpg', 'JPEG')
        page_names.append('media/page'+ str(i) +'.jpg')
    return page_names

# returns the extension of uploaded file
def check_file_type(file):
    file_type = file.split('.')[-1]
    return file_type

# returns all the detected digits in european uploaded file.
# this function is going to be used to give informative message to user
# in the case of there is no detected digits.
def european_prediction(path):
    model = YOLO('hand_written_digits_recognition/yolov8_weights/european/best.pt')
    results = model.predict(source=path, save=True, conf=0.50, line_thickness=2, hide_conf=True)
    detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(model.names[int(c)])
    return detected_classes

# returns all the detected digits in chinese uploaded file.
# this function is going to be used to give informative message to user
# in the case of there is no detected digits.
def chinese_prediction(path):
    model = YOLO('hand_written_digits_recognition/yolov8_weights/chinese/best.pt')
    results = model.predict(source=path, save=True, conf=0.50, line_thickness=2, hide_conf=True)
    detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(model.names[int(c)])
    return detected_classes

# returns all the detected digits in arabic uploaded file.
# this function is going to be used to give informative message to user
# in the case of there is no detected digits.
def arabic_prediction(path):
    model = YOLO('hand_written_digits_recognition/yolov8_weights/arabic/best.pt')
    results = model.predict(source=path, save=True, conf=0.50, line_thickness=2, hide_conf=True)
    detected_classes = []
    for r in results:
        for c in r.boxes.cls:
            detected_classes.append(model.names[int(c)])
    return detected_classes

# used to save the file to media directory, and returns the url of the saved file.
def save_files(file):
    fs = FileSystemStorage()
    filename = file.name
    if ' ' in filename: 
        filename = filename.replace(' ', '_')
    else:
        filename = file.name
    filename = fs.save(filename, file)
    uploaded_file_url = fs.url(filename)
    return uploaded_file_url

# makes the preprocess to the uploaded file.
def preprocess_image(path):
    img = cv2.imread(path)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    img = cv2.resize(img, (640, 640), interpolation = cv2.INTER_AREA)
    cv2.imwrite(path, img)

# only images in the media directory can be displayed to the user
# in this function we copy the prediction image to media directory
def save_predicted_image_to_media_directory(path):
    file_list = path.split('/')
    file_name = file_list[len(file_list) - 1]
    img = Image.open('runs/detect/predict/' + file_name)
    img.save("media/2_" + file_name)
    return "media/2_" + file_name

# function for the index page
def index(request):
    delete_all_previous_predictions_and_saved_images()
    return render(request, "hand_written_digits_recognition/index.html")

# european image upload
def european(request):
    # deletes all the previous images in order to make new predictions
    delete_all_previous_predictions_and_saved_images()
    # form for uploading the file
    form = uploadFileForm()

    # if the submitted method is post, the prediction image will be passed to the django template
    if request.method == 'POST':
        # the default value is false for this variable
        file_is_pdf = False

        # form and file variable, form stores the post values and files values
        # file storest the uploaded file
        form = uploadFileForm(request.POST, request.FILES)
        file = request.FILES['file']
        
        # an empty list
        page_names = []

        # cheks the extension of the uploaded file
        # only .jpg, .jpeg and .pdf extensions are allowed.
        if check_file_type(file.name) not in ['jpeg', 'jpg', 'pdf']:
            return render(request, "hand_written_digits_recognition/prediction.html", {
                "message": "Please upload a jpeg, jpg or pdf file",
                "form": form,
                "type": "EUROPEAN"
            })
        
        # stores the url of the saved file
        uploaded_file_url = save_files(file)

        # if the uploaded file is pdf
        # firstly saves the all pages as images
        # the apply preprocess function to all saved images
        # saves the result of prediction and url of the first page image
        # assigns True to file_is_pdf
        if check_file_type(file.name) == 'pdf':
            page_names = pdf_to_image(uploaded_file_url[1:])
            for page in page_names:
                preprocess_image(page)
            results = european_prediction(page_names[0])
            uploaded_file_url = page_names[0]
            file_is_pdf = True
        else:
            # else just apply preprocess on the image and saves the result of the predictions
            preprocess_image(uploaded_file_url[1:])
            results = european_prediction(uploaded_file_url[1:])

        # if there is no detected digits, returns an informative message to the user
        if len(results) == 0:
            return render(request, "hand_written_digits_recognition/prediction.html", {
                "image": save_predicted_image_to_media_directory(uploaded_file_url),
                "message": "There is no detected digits in the uploaded file.",
                "form": form,
                "type": "EUROPEAN",
                "file_is_pdf": file_is_pdf,
                "total_page": len(page_names),
                "current_page": 1
            })
        
        # else returns the predicted image
        return render(request, "hand_written_digits_recognition/prediction.html", {
            "image": save_predicted_image_to_media_directory(uploaded_file_url),
            "form": form,
            "type": "EUROPEAN",
            "file_is_pdf": file_is_pdf,
            "total_page": len(page_names),
            "current_page": 1
        })
        
    # if the request method is get, form is passed to the django template
    return render(request, "hand_written_digits_recognition/prediction.html", {
        "form": form,
        "type": "EUROPEAN"
    })

# chinese image upload
def chinese(request):
    # deletes all the previous images in order to make new predictions
    delete_all_previous_predictions_and_saved_images()
    # form for uploading the file
    form = uploadFileForm()
    
    # if the submitted method is post, the prediction image will be passed to the django template
    if request.method == 'POST':
        # the default value is false for this variable
        file_is_pdf = False
        
        # form and file variable, form stores the post values and files values
        # file storest the uploaded file
        form = uploadFileForm(request.POST, request.FILES)
        file = request.FILES['file']
        
        # an empty list
        page_names = []

        # cheks the extension of the uploaded file
        # only .jpg, .jpeg and .pdf extensions are allowed.
        if check_file_type(file.name) not in ['jpeg', 'jpg', 'pdf']:
            return render(request, "hand_written_digits_recognition/prediction.html", {
                "message": "Please upload a jpeg, jpg or pdf file",
                "form": form,
                "type": "CHINESE"
            })
        
        # stores the url of the saved file
        uploaded_file_url = save_files(file)

        # if the uploaded file is pdf
        # firstly saves the all pages as images
        # the apply preprocess function to all saved images
        # saves the result of prediction and url of the first page image
        # assigns True to file_is_pdf
        if check_file_type(file.name) == 'pdf':
            page_names = pdf_to_image(uploaded_file_url[1:])
            for page in page_names:
                preprocess_image(page)
            results = chinese_prediction(page_names[0])
            uploaded_file_url = page_names[0]
            file_is_pdf = True
        else:
            # else just apply preprocess on the image and saves the result of the predictions
            preprocess_image(uploaded_file_url[1:])
            results = chinese_prediction(uploaded_file_url[1:])
        
        # if there is no detected digits, returns an informative message to the user
        if len(results) == 0:
            return render(request, "hand_written_digits_recognition/prediction.html", {
                "image": save_predicted_image_to_media_directory(uploaded_file_url),
                "message": "There is no detected digits in the uploaded file.",
                "form": form,
                "type": "CHINESE",
                "file_is_pdf": file_is_pdf,
                "total_page": len(page_names),
                "current_page": 1
            })
        
        # else returns the predicted image
        return render(request, "hand_written_digits_recognition/prediction.html", {
            "image": save_predicted_image_to_media_directory(uploaded_file_url),
            "form": form,
            "type": "CHINESE",
            "file_is_pdf": file_is_pdf,
            "total_page": len(page_names),
            "current_page": 1
        })
        
    # if the request method is get, form is passed to the django template
    return render(request, "hand_written_digits_recognition/prediction.html", {
        "form": form,
        "type": "CHINESE"
    })

# arabic image upload
def arabic(request):
    # deletes all the previous images in order to make new predictions
    delete_all_previous_predictions_and_saved_images()
    # form for uploading the file
    form = uploadFileForm()
    
    # if the submitted method is post, the prediction image will be passed to the django template
    if request.method == 'POST':
        # the default value is false for this variable
        file_is_pdf = False
       
        # form and file variable, form stores the post values and files values
        # file storest the uploaded file
        form = uploadFileForm(request.POST, request.FILES)
        file = request.FILES['file']
        
        # an empty list
        page_names = []

        # cheks the extension of the uploaded file
        # only .jpg, .jpeg and .pdf extensions are allowed.
        if check_file_type(file.name) not in ['jpeg', 'jpg', 'pdf']:
            return render(request, "hand_written_digits_recognition/prediction.html", {
                "message": "Please upload a jpeg, jpg or pdf file",
                "form": form,
                "type": "ARABIC"
            })
        
        # stores the url of the saved file
        uploaded_file_url = save_files(file)

        # if the uploaded file is pdf
        # firstly saves the all pages as images
        # the apply preprocess function to all saved images
        # saves the result of prediction and url of the first page image
        # assigns True to file_is_pdf
        if check_file_type(file.name) == 'pdf':
            page_names = pdf_to_image(uploaded_file_url[1:])
            for page in page_names:
                preprocess_image(page)
                preprocess_image(page)
            results = arabic_prediction(page_names[0])
            uploaded_file_url = page_names[0]
            file_is_pdf = True
            
        else:
            # else just apply preprocess on the image and saves the result of the predictions
            preprocess_image(uploaded_file_url[1:])
            preprocess_image(uploaded_file_url[1:])
            results = arabic_prediction(uploaded_file_url[1:])
        
        # if there is no detected digits, returns an informative message to the user
        if len(results) == 0:
            return render(request, "hand_written_digits_recognition/prediction.html", {
                "image": save_predicted_image_to_media_directory(uploaded_file_url),
                "message": "There is no detected digits in the uploaded file.",
                "form": form,
                "type": "ARABIC",
                "file_is_pdf": file_is_pdf,
                "total_page": len(page_names),
                "current_page": 1
            })
        
        # else returns the predicted image
        return render(request, "hand_written_digits_recognition/prediction.html", {
            "image": save_predicted_image_to_media_directory(uploaded_file_url),
            "form": form,
            "type": "ARABIC",
            "file_is_pdf": file_is_pdf,
            "total_page": len(page_names),
            "current_page": 1
        })
        
    # if the request method is get, form is passed to the django template
    return render(request, "hand_written_digits_recognition/prediction.html", {
        "form": form,
        "type": "ARABIC"
    })

# european pdf page upload
def european_page(request, page_id):

    # for for uploading the file
    form = uploadFileForm()

    # if runs directory exists, removes it before making a new prediction
    isExist = os.path.exists('runs')
    if isExist:
        shutil.rmtree('runs')

    # removes the copied image from the media directory
    for file in os.listdir('media'):
        if file.startswith('2') and file.endswith('jpg'):
            os.remove('media/' + file)

    # stores all the converted image in a list and sorts
    pages = []
    for file in os.listdir('media'):
        if file.startswith('page') and file.endswith('jpg'):
            pages.append(file)

    pages.sort(reverse=False)

    # passes the page id and page name to django template
    if page_id >= len(pages) + 1:
        page_id_2 = len(pages) - 1
        page = pages[len(pages) - 1]

    elif page_id <= 0:
        page_id_2 = 0
        page = pages[0]
    else:
        page_id_2 = page_id
        page = pages[page_id - 1]

    # saves the result of predictions
    results = european_prediction('media/' + page)

    # if there is no detected digits, returns an informative message to the user
    if len(results) == 0:
        return render(request, "hand_written_digits_recognition/prediction.html", {
            "image": save_predicted_image_to_media_directory('media/' + page),
            "message": "There is no detected digits in the uploaded file.",
            "type": "EUROPEAN",
            "file_is_pdf": True,
            "total_page": len(pages),
            "current_page": page_id_2,
            "page":page,
            "form": form
        })
    
    # else returns the predicted image
    return render(request, "hand_written_digits_recognition/prediction.html", {
        "image": save_predicted_image_to_media_directory('media/' + page),
        "type": "EUROPEAN",
        "file_is_pdf": True,
        "total_page": len(pages),
        "current_page": page_id_2,
        "page":page,
        "form": form
    })
    
# chinese pdf page upload
def chinese_page(request, page_id):

    # for for uploading the file
    form = uploadFileForm()

    # if runs directory exists, removes it before making a new prediction
    isExist = os.path.exists('runs')
    if isExist:
        shutil.rmtree('runs')

    # removes the copied image from the media directory
    for file in os.listdir('media'):
        if file.startswith('2') and file.endswith('jpg'):
            os.remove('media/' + file)


    # stores all the converted image in a list and sorts
    pages = []
    for file in os.listdir('media'):
        if file.startswith('page') and file.endswith('jpg'):
            pages.append(file)

    pages.sort(reverse=False)

    # passes the page id and page name to django template
    if page_id >= len(pages) + 1:
        page_id_2 = len(pages) - 1
        page = pages[len(pages) - 1]

    elif page_id <= 0:
        page_id_2 = 0
        page = pages[0]
    else:
        page_id_2 = page_id
        page = pages[page_id - 1]

    # saves the result of predictions
    results = chinese_prediction('media/' + page)

    # if there is no detected digits, returns an informative message to the user
    if len(results) == 0:
        return render(request, "hand_written_digits_recognition/prediction.html", {
            "image": save_predicted_image_to_media_directory('media/' + page),
            "message": "There is no detected digits in the uploaded file.",
            "type": "CHINESE",
            "file_is_pdf": True,
            "total_page": len(pages),
            "current_page": page_id_2,
            "page":page,
            "form": form
        })
    
    # else returns the predicted image
    return render(request, "hand_written_digits_recognition/prediction.html", {
        "image": save_predicted_image_to_media_directory('media/' + page),
        "type": "CHINESE",
        "file_is_pdf": True,
        "total_page": len(pages),
        "current_page": page_id_2,
        "page":page,
        "form": form
    })
    
# arabic pdf page upload
def arabic_page(request, page_id):

    # for for uploading the file
    form = uploadFileForm()

    # if runs directory exists, removes it before making a new prediction
    isExist = os.path.exists('runs')
    if isExist:
        shutil.rmtree('runs')

    # removes the copied image from the media directory
    for file in os.listdir('media'):
        if file.startswith('2') and file.endswith('jpg'):
            os.remove('media/' + file)

    # stores all the converted image in a list and sorts
    pages = []
    for file in os.listdir('media'):
        if file.startswith('page') and file.endswith('jpg'):
            pages.append(file)

    pages.sort(reverse=False)

    # passes the page id and page name to django template
    if page_id >= len(pages) + 1:
        page_id_2 = len(pages) - 1
        page = pages[len(pages) - 1]

    elif page_id <= 0:
        page_id_2 = 0
        page = pages[0]
    else:
        page_id_2 = page_id
        page = pages[page_id - 1]

    # saves the result of predictions
    results = arabic_prediction('media/' + page)

    # if there is no detected digits, returns an informative message to the user
    if len(results) == 0:
        return render(request, "hand_written_digits_recognition/prediction.html", {
            "image": save_predicted_image_to_media_directory('media/' + page),
            "message": "There is no detected digits in the uploaded file.",
            "type": "ARABIC",
            "file_is_pdf": True,
            "total_page": len(pages),
            "current_page": page_id_2,
            "page":page,
            "form": form
        })
    
    # else returns the predicted image
    return render(request, "hand_written_digits_recognition/prediction.html", {
        "image": save_predicted_image_to_media_directory('media/' + page),
        "type": "ARABIC",
        "file_is_pdf": True,
        "total_page": len(pages),
        "current_page": page_id_2,
        "page":page,
        "form": form
    })
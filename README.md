# Hand-written Digit Detector

Hand-written digit recognition project can detect European, Chinese and Arabic hand-written digits which are written with black ink on a white coloured background. This project is developed using Django Framework and Python Programming Language since Django is simple, flexible and reliable. The YOLOv8 algorithm is used for object detection and Roboflow library is used for annotating process since it is able to export the output dataset in the suitable format for YOLOv8.

Some of the challanges in the project are explained below.

1. Since European, Chinese and Arabic digit detectors have seperate links in the project, sometimes, the model tries to detect and classify digits in different types. For example, a Chinese zero can be identified as European digit 3 in European digit detector link.

2. Roboflow does not support some characters. As a result, the Chinese and Arabic digits are classified and translated as European digits. Also it does not support '(' and ')' characters. Not-supported characters turn into '-' character in the output label. For example, the model can figure out the detected object is a Chinese zero, but the label showed to the user '0 -zero-' as a European digit, not in Chinese digit character. So, the website works like a European digit detector and digit translator from Chinese and Arabic to European.

3. Since characters other than the digits are not included in the project, the model tries to detect other possible objects such as letters, which give wrong prediction results. However, the model provides good prediction results when the uploaded file is in requested format (black digit and white background) and just includes the hand-written digits.

4. Multiple digits in the uploaded file can be detected, but they are not combined into a number. The digits in the number are classified seperately. For example, for the number '2023', the user will not be able to see '2023' as a whole number in the label. They will see four seperate labels for each detected digit. E.g. for digit 2, the label value on a label will be seen '2 -two-'.

5. The program supports multi-paged pdf file upload feature. This feature works by saving all the pages in the pdf file as .jpg image file, so this stage takes long time for pdf files with huge number of pages.

This project might be improved by training a single model which contains all the alphabet and ASCII characters, although this may cause wrong predictions on digits with the same representation and different values. The model also could include the words training for classifying words or numbers as a single object.

# How To Install and Run The Poject

1. If the python is not installed on your computer, first install the python.

1.1 Go to https://www.python.org/downloads/ and download the approppriate file for your computer.
1.2 Open the file and follow the steps. Do not forget to click save python.exe file to PATH checkbox in Windows operating systems.
1.3 Click install now
1.4 To check if the installation is successful, write ‘python --version’ to terminal or command prompt. The result should be something like ‘Python 3.11.3’.

2.  If the Anaconda is not installed on your computer, install Anaconda
2.1 Firstly, install Anaconda file from the https://www.anaconda.com
2.2 Complete installation of the Anaconda by clicking ‘Next >’ button. Click save python.exe file to PATH checkbox
2.3 You may need to restart the computer to complete Anaconda download.
2.4 To check if the installation is successful, write ‘conda --version’ to terminal or command prompt. The result should be something like ‘conda 23.3.1’.

3. Create and activate a conda environment
3.1 Create a conda environment by writing ‘conda create --name final_year_project’ on terminal or command prompt.
3.2 Activate the environment by writing ‘conda activate final_year_project’ on terminal or command prompt.

4.  Install necessary libraries
4.1 Install Django by writing ‘conda install Django’ on terminal or command prompt.
4.2 Install opencv by writing ‘conda install -c conda-forge opencv’ on terminal or command prompt. 
4.3 Install ultralytics by writing ‘pip install ultralytics’ on terminal or command prompt.
4.4 Install pdf2image by writing ‘conda install -c conda-forge pdf2image’ on terminal or command prompt. 
4.5 Install poppler by writing ‘conda install -c conda-forge poppler’ on terminal or command prompt. 

5. Change the current directory to the directory where you want to clone the repository on terminal or command prompt.
6. Clone the repository and cd into zna2 folder.
7. Run 'python manage.py runserver'.
8. The website started to run at http://127.0.0.1:8000/ link on localhost. It is just enough to copy and paste this link to access website.

# How To Use The Project?

When the user follows the steps above, they will see an index page which has 3 links. The first link redirects the user to European digit detection link, the second one goes to Chinese digit detection link and the final one redirects to the Arabic digit detection link. When the user clicks on one of them, they will see a page where they can upload an image (.jpeg, .jpg) or pdf file. The user will be able to see the file that they have uploaded, if they chosen an image file. If it is a multi-page pdf file, they will only see the first page of their pdf file. Since the user previews which file is selected, if the wrong file is selected, user can change the file without submitting the wrong file. If the file is correct, they can click on the 'Detect Digits' button to apply model to the uploaded file. If there is a detected digit, user will be able to see its label. If there is no hand-written digits in the image, program will inform user by popping up an alert. If the uploaded file is a pdf file, user can access all the pages in the file. They will be able to see the number of the current page and the total number of pages in the file. Also, the user is able to navigate themselves to previous and next pages in the pdf file by pressing the links under the image. Finally, there is a 'Go Back' button which redirects the user to the index page easily.

# Directory and File Structure of the Project
- hand_written_digits_recognition:
    - datasets
        - arabic_dataset
        - chinese_dataset
        - european_dataset
    - ipynb_files:
        - arabic:
            - yolov8_model_for_arabic_digits.ipynb
        - chinese:
            - yolov8_model_for_chinese_digits.ipynb
        - european:
            - object_localisation:
                - Tensorflow object detection.ipynb
            - object_recognition:
                - Export_European_Digit_Dataset.ipynb
                - Train Model.ipynb
            - yolov8_model_for_european_digits.ipynb
    - results:
        - european: 
            - confusion_matrix.png
            - results.png
            - val_batch0_pred.jpg
        - chinese: 
            - confusion_matrix.png
            - results.png
            - val_batch0_pred.jpg
        - arabic: 
            - confusion_matrix.png
            - results.png
            - val_batch0_pred.jpg
    - static:
        - hand_written_digits_recognition:
            - european.jpeg
            - chinese.jpeg
            - arabic.jpeg
    - templates:
        - hand_written_digits_recognition:
            - index.html
            - prediction.html
    - yolov8_datasets:
        - european_dataset: 
            - mnist-5
        - chinese_dataset: 
            - Chinese-Digits-17
        - arabic_dataset:
            - Arabic-Digits-10
    - yolov8_weights:
        - european:
            - best.pt
            - yolov8s.pt
        - chinese:
            - best.pt
            - yolov8s.pt
        - arabic:
            - best.pt
            - yolov8s.pt
    - urls.py
    - views.py
    - images for testing:
        - european
        - chinese
        - arabic

The images under 'images for testing' folder can be used for testing.
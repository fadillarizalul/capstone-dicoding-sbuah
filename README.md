# capstone-dicoding-sbuah
sBuah is a website application to classify an image of fruit. With this application, user can detect whether the fruit is Fresh or Rotten. By using Machine Learning algorithm, the model results an accuracy of about more than 97%. Model then deployed using Flask on Heroku App. The aim of this project is to reduce fruit waste and largerly food waste that contributes to Climate Change, especially in Indonesia.

## Getting Started  

First of all, ensure that the following requirements already installed on your system.
```text
absl-py==1.0.0
astunparse==1.6.3
cachetools==4.2.4
certifi==2021.10.8
charset-normalizer==2.0.9
click==8.0.3
colorama==0.4.4
Flask==2.0.2
flatbuffers==2.0
gast==0.4.0
google-auth==2.3.3
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.43.0
gunicorn==18.0.0
h5py==3.6.0
idna==3.3
importlib-metadata==4.9.0
itsdangerous==2.0.1
Jinja2==3.0.3
keras==2.7.0
Keras-Preprocessing==1.1.2
libclang==12.0.0
Markdown==3.3.6
MarkupSafe==2.0.1
numpy==1.21.2
oauthlib==3.1.1
opt-einsum==3.3.0
pillow==7.1.2
protobuf==3.19.1
pyasn1==0.4.8
pyasn1-modules==0.2.8
requests==2.26.0
requests-oauthlib==1.3.0
rsa==4.8
six==1.16.0
tensorboard==2.7.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
tensorflow-cpu==2.7.0
tensorflow-estimator==2.7.0
tensorflow-io-gcs-filesystem==0.23.1
termcolor==1.1.0
typing_extensions==4.0.1
urllib3==1.26.7
Werkzeug==2.0.2
wincertstore==0.2
wrapt==1.13.3
zipp==3.6.0
```

This project can be run locally, by use the following command :-  
`python app.py`  
If it doesn't, the local URL would be output in the terminal, just copy it and open it in the browser manually.  
By default, it would be `http://127.0.0.1:5000/`.  
It also deployed onlinely on Heroku, via this [Link](https://sbuah-web-app.herokuapp.com/)  
After that, the webpage should open in the browser automatically.   

Click on `Mari Coba` then `Upload Image` and choose an image from your lcoal to upload.  
Once uploaded, the web-app shall perform result and the output will be displayed.  

## Tools and Stacks  
* Python
* TensorFlow
* Flask
* Heroku

## References  
* https://www.tensorflow.org/tutorials/images/classification
* https://docs.streamlit.io/en/stable/
* https://devcenter.heroku.com/

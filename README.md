# Catheter Prediction AI 🤖
Catheters and tubes are inserted into the lungs to help patients breathe and/or deal with complications in lung surgeries. If they're improperly placed, they can cause throat injury, lung collapse, or even death. Furthermore, these tubes are connected with respirators and used to deal with COVID-19. However, diagnosing if a catheter is properly placed or not is a difficult tak. Experienced doctors must analyze X-rays, which often have multiple catheters in them, closely. Doctors can be inaccurate, and examining x-rays takes valuable time and resources from hospitals already strained by COVID-19.

**Our solution is to use artificial intelligence to solve this problem.** AI is much more accurate than a human and can give quick predictions at any time. Furthermore, it's more accessible to regions without fewer hospitals or other medical support. Our neural network model has over 98.5% accuracy on identifying key medical information about catheter placement in lungs, and was deployed to give predictions on user-uploaded images.

**Scroll down for more information about our solution, links to media/sites, medical context, and more!**

---

# Quick Links
- **[Online Demo](https://catheterdetection.pythonanywhere.com/)**. Because our model is too big to upload to `pythonanywhere.com` - the website host - we unfortunately can't offer online predictions. However, you can experience the UI and get some dummy results.
- **[Video Demo of Actual Model](https://drive.google.com/file/d/1zAZ4V3sclqzgokvNzySCmyiazwHaNH3x/view)**. We hosted our website on a local server that could hold the model weights; here's a video demo of it giving real predictions on a real X-ray.
- **[Presentation Video](https://drive.google.com/file/d/1N2S1eCWgGqcF-_125v8dE7BSYQuR2dgA/view?usp=sharing)**. We present the problem of catheter inaccuracy, medical context, our solution, and a demonstration of a locally-hosted site returning live predictions on a real X-ray.

---

# What do the file directories mean?
<pre>
<strong>main /</strong>
   <strong>deployment /</strong>                                  #code for deploying the model
      <strong>app.py</strong>                                     #the flask application for our website.
      <strong>app_demo_site.py</strong>                           #the modified flask application for the demo site (https://catheterdetection.pythonanywhere.com); since the host cannot hold our model, the code was modified with dummy data to provide a UI experience nonetheless.
      <strong>templates /</strong>                                #a folder for html and css designs for pages
         <strong>index.html</strong>                              #the home page
         <strong>prediction.html</strong>                         #the prediction page
   <strong>model /</strong>                                       #code for building and testing the model
      <strong>ranzcr-clip-efficientnet-initial.ipynb</strong>     #this was our initial baseline model; diff. model structure and rudimentary augmentation.
      <strong>ranzcr-clip-efficientnet-final.ipynb</strong>       #this is our final model solution; five-folds, more advanced structure, and sophisticated augmentationl.
</pre>


---

---

# Navigate
Helpful links to jump to a particular section.
- [Catheter Prediction AI 🤖](#catheter-prediction-ai---)
- [Quick Links](#quick-links)
- [What do the file directories mean?](#what-do-the-file-directories-mean-)
- [Navigate](#navigate)
- [Approach](#approach)
  * [Modeling Data with Deep Learning](#modeling-data-with-deep-learning)
    + [Model](#model)
    + [Preprocessing and Augmentation of Data](#preprocessing-and-augmentation-of-data)
    + [Technologies and Hardware](#technologies-and-hardware)
  * [Deployment to Website](#deployment-to-website)
- [Understanding the Data and Medical Context](#understanding-the-data-and-medical-context)
  * [Different Types of Catheters](#different-types-of-catheters)
  * [Image Data](#image-data)
  * [Targets](#targets)
- [The Story](#the-story)
  * [Inspiration](#inspiration)
  * [Challenges](#challenges)
  * [What We Learned](#what-we-learned)
  * [Next Steps for Catheter Recognition AI](#next-steps-for-catheter-recognition-ai)
 
---

# Approach

## Modeling Data with Deep Learning
### Model
- Transfer learning on an EfficientNetB5 model with preloaded ImageNet weights, then fine-tuned on dataset. 
- Images are cropped to a `512x512` size.
- Adam optimizer used; 2 epochs of warm-up learning rate and exponential decay for 13 epochs.
- Used `MultilabelStratified` 5-fold cross validation strategy. (Five identically-structured models were trained on 80% of the data each; final result is a weighted average of the predictions of each model.)
- Performance: 98.5 accuracy and 95.9 [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)) on test data, 97.1 [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for train data (with augmentation).

### Preprocessing and Augmentation of Data
- Dataset (`.jpg` and `.png` images) converted into a TensorFlow datasets format (`tf.dataset`) for quick deep learning processing.
- Augmentations were implemented using TensorFlow matrix multiplication.
   - 10-degree range random rotation left or right.
   - Random shear within range of 5%.
   - Random height-wise zoom within range of 5%.
   - Random width-wise zoom within range of 5%.
   - Random height-wise shift within range of 5%.
   - Random width-wise shift within range of 5%.
```python
# our sample code for applying a rotation to an image using TensorFlow matrix multiplication.
c1   = tf.math.cos(rotation)
s1   = tf.math.sin(rotation)
one  = tf.constant([1],dtype='float32')
zero = tf.constant([0],dtype='float32')

rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                             -s1,  c1,   zero, 
                             zero, zero, one])  
                            
# sample code for shear
c2 = tf.math.cos(shear)
s2 = tf.math.sin(shear)    

shear_matrix = get_3x3_mat([one,  s2,   zero, 
                          zero, c2,   zero, 
                          zero, zero, one]) 
```

Examples of augmented X-rays:

![](https://raw.githubusercontent.com/andre-ye/catheter-prediction-app/main/x-ray-augmented.png)

### Technologies and Hardware
- Hardware: Nvidia Telsa P100 TPU v3-8 with 8 cores.
- Language: Python 3.8, model developed in Jupyter Notebooks.
- Linear algebra processing with NumPy.
- Data labels and directories manipulated with Pandas.
- Neural network developed in Keras/TensorFlow.

## Deployment to Website
- Deployed using Flask in Python.
- Hosted by `pythonanywhere.com`.
- Used HTML and CSS to create pages to display results.
- Loaded Keras/TensorFlow neural network architecture and weights, and returned prediction for user-uploaded image of x-ray.

---

# Understanding the Data and Medical Context
## Different Types of Catheters
There are 4 major types of catheters that are placed in patients to assist with breathing and our model can simultaneously detect abnormalities in all 4 categories if present in one single X-ray image.

| Type | Description |
| --- | --- | 
| Endotracheal Tube (ETT) | The ETT tube is placed through the mouth, the tube is then connected to ventilator that delivers oxygen to the patient. |
| Nasogastric Tube (NGT) | The NGT tube is placed through the nose down to the stomach. A syringe is usually connected at the other end to extract the needed content. |
| Central Venous Catheter (CVC) | The CVC is placed through a large vein, needed in patients are more generally more ill. Can be used to give medicine. |
| Swan-Ganz Catheter | This is used for a process called the right heart catheterization. It is mainly used to detect heart failures, monitor therapy, and evaluate the effect of certain drugs. |

## Image Data
X-rays of the lung with any combination of ETTs, NGTs, CVCs, or Swan Gantz Catheters present.

![](https://raw.githubusercontent.com/andre-ye/catheter-prediction-app/main/x-ray-images.png)

## Targets
This was a binary multilabel multiclass problem - there are multiple targets, each image can have multiple targets, and each target is either `0` or `1`.. Our model needed to predict 11 targets:

| Target | Description |
| --- | --- |
| `ETT - Abnormal` | The ETT is present and abnormally placed. |
| `ETT - Borderline` | The ETT is present and may be incorrectly placed. |
| `ETT - Normal` | The ETT is present and normally placed. |
| `NGT - Abnormal` | The NGT is present and abnormally placed. |
| `NGT - Borderline` | The NGT is present and may be incorrectly placed. |
| `NGT - Normal` | The NGT is present and normally placed. |
| `NGT - Incorrectly Imaged` | The x-ray has been incorrectly imaged. |
| `CVC - Abnormal` | The CVC is present and abnormally placed. |
| `CVC - Borderline` | The CVC is present and may be incorrectly placed. |
| `CVC - Normal` | The CVC is present and normally placed. |
| `Swan Ganz Catheter Present` | The Swan Ganz Catheter is present. |

---

# The Story
## Inspiration
This project was inspired by data from the Royal Australian and New Zealand College of Radiologists (RANZCR), where radiographs are analyzed to correct the positioning of catheters in a patient's lungs. X-rays are easily taken, but the position of catheters can be difficult to assess. With COVID-19, respirators and catheters are in higher demand than ever, straining hospital resources. AI can help take the place of valuable doctors and radiologists during this time by providing more accurate and accessible judgements on catheter predictions. They're also more accessible to regions with less medical support. By freeing up medical personell and making catheters safer to use, we believe our model can help aid the COVID crisis.

## Challenges
- **Model too large to host.** Just one of our five models is 300-400 MB large, and our online deployment service, `pythonanywhere.com`, can only host 100 MB for one directory. We ended up using our website (`https://catheterdetection.pythonanywhere.com/`) as a UI experiment with dummy code.
- **Large dataset.** The dataset consists of over 40k images, meaning it is a large amount of data - about 12 GB. This means that we needed to convert it into a highly specialized from - TensorFlow datasets - and had to be careful about applying augmentations or other changes to the images. This also made our experimentation slower.
- **Hardware availabilities.** We use Kaggle and Google Colab's TPU and GPU, which have time limits.
- **Not enough time.** Deep learning models take a long time to train, so to find the true best model, we'd need a bit more time.

## What We Learned
Some of our key learnings:
- **How to deal with big datasets.** This dataset was particularly large, so we had to learn how to make smart decisions to maneuver around it.
- **How to deploy a deep learning model w/ Flask.** This was our first time using Flask to deploy a model (prior we had experience with Django).
- **How to manipulate hardware.** We had to configure our TPU and GPU for optimal training speed and experimentation w/ model structure.

## Next Steps for Catheter Recognition AI
- Model deployment on live website, being able to predict online from user inputed images.
- Further improving the model by adding complexity and augmentation.
- Designing a more user friendly UI interface.
- Training with annotated images.
- Adding regularization by increasing the drop connect rate in model structure.

---

[Back to top](#)


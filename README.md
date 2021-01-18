# Catheter Prediction AI 🤖
Catheters and tubes are inserted into the lungs to help patients breathe and/or deal with complications in lung surgeries. If they're improperly placed, they can cause throat injury, lung collapse, or even death. Furthermore, these tubes are connected with respirators and used to deal with COVID-19. However, diagnosing if a catheter is properly placed or not is a difficult tak. Experienced doctors must analyze X-rays, which often have multiple catheters in them, closely. Doctors can be inaccurate, and examining x-rays takes valuable time and resources from hospitals already strained by COVID-19.

**Our solution is to use artificial intelligence to solve this problem.** AI is much more accurate than a human and can give quick predictions at any time. Furthermore, it's more accessible to regions without fewer hospitals or other medical support. Our neural network model has over 98.5% accuracy on identifying key medical information about catheter placement in lungs, and was deployed to give predictions on user-uploaded images.

**Scroll down for more information about our solution, links to media/sites, medical context, and more!**

---

# Quick Links
- **[Online Demo](https://catheterdetection.pythonanywhere.com/)**. Because our model is too big to upload to `pythonanywhere.com` - the website host - we unfortunately can't offer online predictions. However, you can experience the UI and get some dummy results. 🙂
- 

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
- [Approach](#approach)
  * [Modeling Data with Deep Learning](#modeling-data-with-deep-learning)
    + [Model](#model)
    + [Preprocessing and Augmentation of Data](#preprocessing-and-augmentation-of-data)
    + [Training Strategies](#training-strategies)
    + [Technologies and Hardware](#technologies-and-hardware)
  * [Deployment to Website](#deployment-to-website)
- [Understanding the Data and Medical Context](#understanding-the-data-and-medical-context)
  * [Different Types of Catheters](#different-types-of-catheters)
- [The Story](#the-story)
  * [Inspiration](#inspiration)
  * [Challenges](#challenges)
  * [What We Learned](#what-we-learned)
  * [Next Steps for Catheter Recognition AI](#next-steps-for-catheter-recognition-ai)
 
---

# Approach

## Modeling Data with Deep Learning
### Model
- Transfer learning on an EfficientNetB6 model with preloaded ImageNet weights, then fine-tuned on dataset. 
- Images are cropped to a `512x512` size.
- Adam optimizer used; 2 epochs of warm-up learning rate and exponential decay for 13 epochs.
- Used `MultilabelStratified` 5-fold cross validation strategy. (Five identically-structured models were trained on 80% of the data each; final result is a weighted average of the predictions of each model.)

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
```

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
This project was inspired by data from the Royal Australian and New Zealand College of Radiologists (RANZCR), where radiographs are analyzed to correct the positioning of catheters in a patient's lungs. X-rays are easily taken, but the position of catheters can be difficult to assess. With COVID-19, respirators and catheters are in higher demand than ever, straining hospital resources and 

This project was inspired by the works of Royal Australian and New Zealand College of Radiologists(RANZCR), where they anaylze radiographs to correct the position of Catheters. Xrays are easily taken but the position of Cateters placed are hard to reconize and detect errors, while radiologists can anaylze radiograph but there are not enough people that's qualified and resources are scarce. From the data that was [provided by RANZCR](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data), we trained our model to automate the process of spotting errors in placement of Catheter tubes. This could possibly help the COVID crisis since Catheters are commonly used to assist with patient breathing. 

## Challenges
Model too large to host, etc.
Having to deal with the large amount of data (12G~).
Hardware avaliabilities.
Not enough time to complete under the 3 days time contraint.

## How we built it
We used the data provided by RANZCR, around 40k images. Using machine learning, we are able to predict images faster and more accurate than humans. We used the popular machine learning framework Tensorflow and built a Convolutional Neural Network(CNN), Specifically, we used the EffcientNetB5 structure. We spend time preproccessing the images, applying augmentations so our model will be more robust on unseen datasets. Some examples of the augmentations that we applied are zoom, shear and rotation. Being robust and being able to predict well on unseen images is crucially important, thus we trained 5 models on different parts of the datasets with overlaps (Kfold validation), ensuring the robustness. For each model we trained it for 5 iterations, obtaining result of around 98.5 accuracy (95.9 [ AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)). Finally, we built a website using the Flask framework for a UI system and the ability to recive user inputs. 

## What We Learned
## Next Steps for Catheter Recognition AI

---

[Back to top](#)


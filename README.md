# Catheter Prediction AI 🤖

Catheters and tubes are inserted into the lungs to help patients breathe and/or deal with complications in lung surgeries. If they're improperly placed, they can cause throat injury, lung collapse, or even death. Furthermore, these tubes are connected with respirators and used to deal with COVID-19. However, diagnosing if a catheter is properly placed or not is a difficult tak. Experienced doctors must analyze X-rays, which often have multiple catheters in them, closely. Doctors can be inaccurate, and examining x-rays takes valuable time and resources from hospitals already strained by COVID-19.

Our solution is to use artificial intelligence to solve this problem. AI is much more accurate than a human and can give quick predictions at any time. Furthermore, it's more accessible to regions without fewer hospitals or other medical support. Our neural network model has over 98.5% accuracy on identifying key medical information about catheter placement in lungs, and was deployed to give predictions on user-uploaded images.

---

# Quick Links
- **[Online Demo](https://catheterdetection.pythonanywhere.com/)**. Because our model is too big to upload to `pythonanywhere.com` - the website host - we unfortunately can't offer online predictions. However, you can experience the UI and get some dummy results. 🙂
- [Hello]().

---

---

# Navigate
(Editors: visit [here](https://ecotrust-canada.github.io/markdown-toc/) to generate a table of contents for markdown code.)

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
- Efficientnet with global pooling, insert diagram

We used Neural Networks to approach the probelm, specifically, a Convolutional Neural Network (CNN). We used transfer learning on EffcientNetB6 with preloaded imagenet weights. The images were croped to 512x512 size, it's preprocessed with Rotation, Shear, Zoom and Shift augmentations. We trained our model with the Adam optimizer with 2 epochs of warm up learning rate, then it exponentially decreases for 13 more epochs. We used Tensor Processing Units(TPU) to accelerate the training process, taking about 4~ hours. We ensured the accuracy of our model on test data by using K fold cross validation strategy, 5 fold MultilabelStratified K fold was implemented. The model was coded in python using the tensroflow/keras framework.

### Preprocessing and Augmentation of Data
- Rotations
- tensorflow datasets (can insert some code snippets)
```python
def sample_code():
 like_maybe_the_loading_function()
```

### Training Strategies
- Discuss five-fold

### Technologies and Hardware
- TPU

## Deployment to Website
= TensorFlow

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

---

# The Story
## Inspiration
## Challenges
## What We Learned
## Next Steps for Catheter Recognition AI

---


[Back to top](#)


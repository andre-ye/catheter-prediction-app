# Catheter Prediction AI 🤖

Catheters and tubes are inserted into the lungs to help patients breathe and/or deal with complications in lung surgeries. If they're improperly placed, they can cause throat injury, lung collapse, or even death. Furthermore, these tubes are connected with respirators and used to deal with COVID-19. 

However, diagnosing if a catheter is properly placed or not is a difficult tak. Experienced doctors must analyze X-rays, which often have multiple catheters in them, closely. Doctors can be inaccurate, and examining x-rays takes valuable time and resources from hospitals already strained by COVID-19.

Our solution is to use artificial intelligence to solve this problem. AI is much more accurate than a human and can give quick predictions at any time. Furthermore, it's more accessible to regions without fewer hospitals or other medical support. Our neural network model has over 98.5% accuracy on identifying key medical information about catheter placement in lungs, and was deployed to give predictions on user-uploaded images.

## Quick Links
- **[Online Demo](https://catheterdetection.pythonanywhere.com/)**. Because our model is too big to upload to `pythonanywhere.com` - the website host - we unfortunately can't offer online predictions. However, you can experience the UI and get some dummy results. 🙂

Build description off this
https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data

## Understanding the Data
#### Different Types of Catheters
There are 4 major types of catheters that are placed in patients to assist with breathing and our model can simultaneously detect abnormalities in all 4 categories if present in one single X-ray image.

- Endotracheal Tube (ETT)
  - The ETT tube is placed through the mouth, the tube is then connected to ventilator that delivers oxygen to the patient.
- Nasogastric Tube (NGT)
  - The NGT tube is placed through the nose down to the stomach. 
  - A syringe is usually connected at the other end to extract the needed content
- Central venous catheter (CVC)
  - This is a catheters that's placed through a large vein, this is needed in patients are more generally more ill.
  can be used to give medicine.
- Swan-Ganz catheter
  - This is used for a process called the right heart catheterization, it's mainly used to detect heart failures, monitor therapy and evaluate the effect of certain drugs.

## Process


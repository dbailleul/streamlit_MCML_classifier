# Multi-Class Multi-Level Classification Model on Streamlit

![image](https://github.com/dbailleul/streamlit_MCML_classifier/streamlit_MCML_demo.jpg)

This is the free version of an demo I've made at work, with:
  *40,779 images from [Planet dataset](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) from Kaggle competition in 2017
  *Transfer Learning with [VGG16-model](https://keras.io/api/applications/)
  *Fine-Tuning with four last layers retrained (VGG16 has 1000 classes and Planet dataset 17 classes)
  *Data augmentation
  *F2-value of 0.896

The app can be found here:
[https://appmcmlclassifier-emqqvjatxqz7hpzlnnqdel.streamlit.app/](https://appmcmlclassifier-emqqvjatxqz7hpzlnnqdel.streamlit.app/)

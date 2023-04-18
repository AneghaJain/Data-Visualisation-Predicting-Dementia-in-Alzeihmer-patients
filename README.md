# Predicting-Dementia-in-Alzeihmer-patients

This project is a combination of different machine learning,deep learning and ensemble learning models that predict the onset of dementia in patients with Alzheimer's disease. The models are trained on a dataset of patient information, including demographic information, cognitive test scores, and clinical assessments.

##Dataset
The dataset used in this project can be found on the Kaggle website. It contains data from patients diagnosed with Alzheimer's disease, including demographic information, medical history, and cognitive assessments.

##Data Cleaning and Preprocessing
The dataset requires extensive cleaning and preprocessing before it can be used for machine learning. The notebook contains the code for cleaning and preprocessing the data, including handling missing values and converting categorical variables to numerical.

##Exploratory Data Analysis (EDA)
EDA is a crucial step in any machine learning project. The notebook contains the code for exploratory data analysis, including visualizations and statistical analysis.

##Feature Extraction
The project uses MRMR(Minimum Redundancy Maximum Relevance) technique for feature extraction and the code for this is also provided in the notebook.We have run a deep learning model with MRMR multiple times,reducing one feature at every iteration to find the optimal number of features and maximum accuracy.

Machine Learning Model
The notebook uses scikit-learn and keras to build a 9 model-MLP,LSTM,CNN,AdaBoost,GradientBoost,XGBoost,RandomForest,SVM,KNN to predict dementia in Alzheimer's patients. The notebook includes code for data splitting, model training, and evaluation.

Acknowledgments
Kaggle for providing the dataset used in this project.
Scikit-learn, Pandas, Numpy, Matplotlib,Keras and Seaborn for their valuable contributions to the project.

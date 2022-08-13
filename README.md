# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This dataset contains data about possible bank loans applicants based on a marketing campaign. In this task the goal was to develop a model that after being trained with the data on each individual will be able to pretict whether the customer will subscribe to the service. This will be done after exploring two different approaches: one model using a hpyerparameter-optimized logistic regression model and another model that was built using AutoML

Based on the primary metric of AUC, the best performing model was the MaxAbsScaler, LightGBM with a AUC weighted of 0.94859 which was a result of using AutoML.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
This sections explains the Scikit-learn Pipeline which was used to run an automated machine learning experiment using SDK. When using the SDK, you have greater flexibility, and you can set experiment options using the AutoMLConfig class. 
Steps include:

Data for Trainining:
The data was loaded in using the URL that was provided in the exercise. Upon successful loading of the data then followed data cleaning such as removing missing values (NA's) and one hot incoding. Once the cleaning was done the data was split into test and training data using the train_test_split with 30% being reserved for testing. The model was then fitted. The Logistic Regression model which is a classifier  was used in this instance to predict column "y". The parameter used in the training script were the regularization strength and maximum number if iterations parsed as arguments. 

Other steps included Creating a ScriptRunConfig Object to specify the configuration details of your training job and then a HyperDriveConfig using the src object, hyperparameter sampler, and policy. 

Hyperparameter Tuning:
The Azure's Hyperdrive service was used for hyperparameter tuning. The HyperDriveConfig had the following attributes:
referencing the run_config, hyperparameter_samplinand policy
primary_metric_name='AUC',
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
max_total_runs=4,
max_concurrent_runs=4)


Parameter sampling:
For this task i utilized the RandomParameterSampling where the hyperparameters are selected ramdonly from the search space. I used this method because it is the fastest option and supports early termination of low-performance runs. This is key to save resources and to ensure alot of time is not spent on the runs. One of the key benefits of random sampling is that:
-It does not require pre-specified values (like grid search)
-Makes full use of all available nodes (unlike bayesian parameter sampling)

Early termination policy:
Early termination improves computational efficiency. I opted the Bandit policy which is the fastest. Bandit policy is based on slack factor/slack amount and evaluation interval and terminates the run where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. Other key benefits of this policy are:
-More flexibility than truncation and median stopping.

## AutoML

The AUtoML task involved creating a TabularDataset using TabularDatasetFactory and cleaning the data this was then followed by calling on to the AutoMLConfig specifying the following parameters:
experiment_timeout_minutes=30,
    task= 'classification',
    primary_metric= 'AUC',
    training_data=ds,
    label_column_name=y,
    n_cross_validations=5)

The model produced by the AutoML was a MaxAbsScaler, LightGBM, 

The Hyperparameters attained from the AutoML were as follows:
"class_name": "LightGBMClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "boosting_type": "goss",
        "colsample_bytree": 0.7922222222222222,
        "learning_rate": 0.026323157894736843,
        "max_bin": 110,
        "max_depth": -1,
        "min_child_weight": 8,
        "min_data_in_leaf": 0.00001,
        "min_split_gain": 0.7894736842105263,
        "n_estimators": 200,
        "num_leaves": 56,
        "reg_alpha": 0.894736842105263,
        "reg_lambda": 0.10526315789473684,
        "subsample": 1
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

When using the SDK, you have greater flexibility, and you can set experiment options using the AutoMLConfig class. Some of the considerable differences betwwen the two is that for AutoML:
-

As stated in the summary, the AutoML resulted in a better solution was the most efficient as it does not involve most of the steps used in the scikit-learn pipeline. Both models were measured with the AUC. With the AutoML model with a AUC weighted of 0.94859 and that of the Scikitlearn with an AUC of 0.91403. The difference was slight. In terms of the architecture, to start with the same data was used in both pipelines and the same cleaning tasks but the automl pipeline tests a number of scalers in combination with models and adds preprocessing step prior to model training. The difference was in the architecuture as a logistic regression from the scikit-learn pipeline uses a fitted logistic function with a threshold to carry out binary classification whilst as i already mentioned a automl pipeline tests and averages different scalers to reach a final prediction.


## Future work
In the future i believe running the automlexperiment for a much longer time (more than 30min) would result in a more effective model although it will be at the expense of paying for resources. Also as an improvement there can be an increase of the number of iterations which will allow the pipeline to run through more models.

In terms of the dataset, a more balanced dataset can be used. The dataset if unbalanced affects the performance of the models (which is why i didnt use metrics such as accuracy) and could possibly introduce bias to be model. Ensuring fairness to the data is key so that the model does not discriminate against one group/feature to another.

## Proof of cluster clean up
The cluster was deleted for clean up and to avoid overspending resources.



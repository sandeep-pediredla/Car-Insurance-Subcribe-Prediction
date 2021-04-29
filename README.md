# Car Insurance Subcribe Prediction:  
## Description
This is a dataset from one bank in the United States. Besides usual services, this bank also provides car insurance services. The bank organizes regular campaigns to attract new clients. The bank has potential customers’ data, and bank’s employees call them for advertising available car insurance options. We are provided with general information about clients (age, job, etc.) as well as more specific information about the current insurance sell campaign (communication, last contact day) and previous campaigns (attributes like previous attempts, outcome).

You have data about 4000 customers who were contacted during the last campaign and for whom the results of campaign (did the customer buy insurance or not) are known.

## Requirement:
The purpose of a marketing campaign is to enrol new customers or upsell to existing ones. 

### ML Task
This is a classification task that consists of predicting whether a given customer will buy or not a car insurance from a campaign ran by a bank.

## Expected for assignment:
What is expected from this challenge:
- Perform an exploratory data analysis (EDA) of the dataset, including data cleaning, and visualisations as needed.
- Train and evaluate a ML model for the task presented using one of the following frameworks:
XGBoost, Scikit-Learn, TensorFlow or PyTorch. (We know there are notebooks in Kaggle, please don’t use them, we want to know how YOU would solve this challenge.) 
- Create REST service that will use trained model for real-time decision making. This REST service should be functional. Pay attention to API design. Document all your assumptions and if you decide to take shortcuts please document why you are taking them and how it should be done in production-ready environment- Properly document and explain all assumptions and decisions to determine why you choose a given approach for this challenge. We would love to learn what is your thinking at each step while building a predictive model.

# Solution: 

## Exploratory data analysis:

Understanding of data, find patterns within the data & basic high level insights into data.
If a ML model efficeny changed, in this case EDA helps by having a quick look what changed in data.

We have used Jupyter Notebook for Exploratory data analysis.

#### Pipeline:
![Alt text](docs/images/pipeline_flow.jpg")

Load Data
Handle Missing value
Removing duplicates
Outlier Treatment
Normalizing and Scaling
Encoding
Extract Features
Train Model (We have used Scikit-Learn as a machine learning library) 
Predict
Expose the model as a web service

## API Architecture:
Develop REST API that allow us to predict customer's chance of buying policy. 
Endpoint services:
Predict
Train (optional): for this we can use toggle operation at service level

Deployment:
We were wrapping rest services inside docker.

To achieve scalability:
From "Load Data" to "Train Model" steps, we should use distributed preocessing engine like Spark, Flink, Apex, etc.. and saves train model to S3 or Hdfs, etc..

One important consideration, we need to take is saving the trained model. This should be save on a distributed datastore like HDFS, S3, etc... Thereby based on the load we can use Kubernetes or mesos to scale up or down the rest service which loads the trained model from S3/Hdfs and starts serving frontend executives.

Issues:
For Encoding, I have used one hot encoder technique which is highly dependent on the type of data. To most extent, I Have tried to reduce impact of it like gender, month, outcomes, etc.. But even then some fields are out of control like job details, so if there is a new job thats not they in training set the prediction may have some impact.

Handling of NA, empty & missing depends on the data.

#### Improvements:
1. Use a distributed preocessing engine instead of Scikit-Learn
2. Read datasets from external source

## Development:

#### Building project:
Command to build docker

```sh
docker build -t car_insurance_prediction .
```
docker run -p 8888:8888 car_insurance_prediction


## License

Public
**Free Software, Hell Yeah!**

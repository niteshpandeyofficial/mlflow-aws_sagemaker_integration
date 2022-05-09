import os
import mlflow
import argparse
import numpy as np
import pandas as pd 
import mlflow.sklearn 
from sklearn import datasets
from from_root import from_root
from urllib.parse import urlparse
from utils.sagemaker_integration import upload
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error ,mean_squared_error,r2_score


def get_data():
    try:
        boston = datasets.load_boston()
        features = pd.DataFrame(boston.data,columns=boston.feature_names)
        targets = boston.target
        return features,targets
    
    except Exception as e:
        raise e

def evaluate(y_test,predict):
    rmse=np.sqrt(mean_squared_error(y_test,predict))
    mae=mean_absolute_error(y_test,predict)
    r2=r2_score(y_test,predict)

    return rmse,mae,r2

def main(max_depth,n_estimators,min_samples_split):
    features,targets=get_data()
    x_train,x_test,y_train,y_test=train_test_split(features,targets,random_state=123,test_size=0.30)

 #   uri="http://127.0.0.1:5000"
 #   mlflow.set_tracking_uri(uri)
    with mlflow.start_run():
        rf=RandomForestRegressor()
        rf.fit(x_train,y_train)
        predict=rf.predict(x_test)

        rmse,mae,r2=evaluate(y_test,predict)
        
        # print(f'elasticnet parameter : alpha:{alpha} , l1_ratio:{l1_ratio}')
        # print(f'elasticnet metrics : rmse:{rmse} , mae:{mae},r2:{r2}')
        
        mlflow.log_param("max_depth",max_depth)
        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("min_samples_split",min_samples_split)

        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)
        mlflow.sklearn.log_model(rf,"model")

        try:
            inp=input("Push Model to S3(Y or N):")
            print(inp)
            if inp=='Y':
                print('before path')
                runs=os.path.join(from_root(),'artifacts/')
                #runs='D:\\aws_document_practlicals\\mlopwithaws\\mlruns'
                print('after path')
                print(runs)
                print('Path to mlrus exists: ',os.path.exists(runs))
                status=upload(s3_bucket_name='mlflow-demo-testing',ml_run_direc=runs)
                print(status)
        
        except Exception as e:
            return f"Error occurred :{e.__str__()}"




if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument("--max_depth","-d",type=float,default=29)
    args.add_argument("--n_estimators","-n",type=float,default=50)
    args.add_argument("--min_samples_split","-s",type=float,default=5)
    parsed_args=args.parse_args()
    try:
        main(max_depth=parsed_args.max_depth,n_estimators=parsed_args.n_estimators,min_samples_split=parsed_args.min_samples_split)
    except Exception as e:
        raise e

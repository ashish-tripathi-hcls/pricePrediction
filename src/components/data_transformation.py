from sklearn.impute import SimpleImputer #Handling Missing Values
from sklearn.preprocessing import StandardScaler #Handling feature Scaling
from sklearn.preprocessing import OrdinalEncoder #Ordinal encoding
import os

#pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_objct

##data transformation config

@dataclass
class TransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')




## data ingestion config class
class DataTransformation:
   def __init__(self):
       self.data_transformation_config=DataTransformation()



   def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')
            #Segregating ordinal and scalar encoded columns
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

            #define custom ranking for ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline initiated')
            #numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('Ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scalar',StandardScaler())
                ]
            )

            preproccesor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
    
            return preproccesor
            logging.info('Pipeline completed')
    
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)

   def initiate_data_transformation_object(self,train_path,test_path):
       
        try:
            #reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train test data complete')
            logging.info(f'train dataframe head : \n{train_df.head().to_string()}')
            logging.info(f'test dataframe head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            ## diving features into independent and dependent feature


            input_features_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_features_train_df=train_df[target_column_name]

            input_features_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_features_test_df=test_df[target_column_name]

            ##apply the transformation
            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            logging.info('Applying preprocessing object on training and test datasets. ')

            train_arr=np.c_[input_feature_train_arr,np.array(target_features_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_features_test_df)]

            save_objct(

                file_path=self.data_transformation_config.preprocessor_ob_file_path,obj=preprocessing_obj
            )
            logging.info('preprocessor pickle is created')
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)






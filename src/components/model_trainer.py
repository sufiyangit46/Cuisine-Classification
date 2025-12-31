import pandas as pd
import sys

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class ModelTrainer:
    def train(self,data_path,model_path):
        try:
            logging.info('Loading data...')
            df=pd.read_csv('S:RestaurantData.csv')
            logging.info('Data loaded')

            df.dropna(subset=['Cuisines'],inplace=True)

            df['Cuisines']=df['Cuisines'].apply(lambda x:x.split(',')[0])

            cuisine_counts=df['Cuisines'].value_counts()
            valid_cuisines=cuisine_counts[cuisine_counts>10].index

            df=df[df['Cuisines'].isin(valid_cuisines)]

            features=[
                'City',
                'Average Cost for two',
                'Price range',
                'Has Table booking',
                'Has Online delivery',
                'Aggregate rating',
                'Votes'
            ]

            X=df[features]
            y=df['Cuisines']

            categorical_features=[
                'City',
                'Has Table booking',
                'Has Online delivery',
            ]

            numerical_features=[
                'Average Cost for two',
                'Price range',
                'Aggregate rating',
                'Votes'
            ]

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features),
                    ('num',StandardScaler(),numerical_features),

                ]
            )

            model=RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )

            pipeline = Pipeline(
                steps=[
                    ('preprocessing',preprocessor),
                    ('model',model)
                ])

            X_train, X_test, y_train, y_test = train_test_split(
                X,y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            logging.info('Training model...')
            pipeline.fit(X_train,y_train)

            y_pred=pipeline.predict(X_test)
            y_proba=pipeline.predict_proba(X_test)

            accuracy=accuracy_score(y_test,y_pred)
            logging.info(f'Model accuracy is {accuracy}')

            macro_f1=f1_score(y_test,pipeline.predict(X_test),average='macro')
            logging.info(f'Macro F1 score is {macro_f1}')

            top3_accuracy=top_k_accuracy_score(y_test,y_proba,k=3,labels=pipeline.classes_)
            logging.info(f'Top 3 accuracy is {top3_accuracy}')

            save_object(model_path,pipeline)
            logging.info('Model saved')

        except Exception as e:
            raise CustomException(e,sys)




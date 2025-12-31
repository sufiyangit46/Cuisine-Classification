from src.components.model_trainer import ModelTrainer
from src.logger import logging

DATA_PATH='S:RestaurantData.csv'
MODEL_PATH='model/cuisine_pipeline.pkl'

if __name__=='__main__':
    logging.info('Training pipeline')
    trainer = ModelTrainer()
    trainer.train(DATA_PATH,MODEL_PATH)
    logging.info('Training completed')

from build_model import build_model
from etl import bbc_data_pipeline

data, target = bbc_data_pipeline()
model = build_model()
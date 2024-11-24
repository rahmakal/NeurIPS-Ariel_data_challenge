import pandas as pd
from src.model import predict
from src.preprocess import preprocess_signals
from src.model import submit

dataset = "test"

axis_info = pd.read_parquet("data/axis_info.parquet")
preprocessed_signal = preprocess_signals(dataset)
print(preprocessed_signal.shape)

predictions_spectra = predict(preprocessed_signal)

submit(predictions_spectra)

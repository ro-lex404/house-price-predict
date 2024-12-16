import pandas as pd

obj = pd.read_pickle(r"C://Users/alexe/Desktop/house-price-predict/land_price_prediction_model.pkl")

file = "myfile.txt"

with open(file,"w") as myfile:
    myfile.write(str(obj))

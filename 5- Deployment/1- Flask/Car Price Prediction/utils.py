import json
import pickle
import numpy as np

car_models = None
data_columns = None
model = None


def predict_price(name, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, age):
    try:
        loc_index = data_columns.index(name.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = km_driven
    x[1] = fuel
    x[2] = seller_type
    x[3] = transmission
    x[4] = owner
    x[5] = mileage
    x[6] = engine
    x[7] = max_power
    x[8] = seats
    x[9] = age
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)


def get_car_model_names():
    return car_models


def load_saved_artifacts():
    print("Loading artifacts")
    global car_models
    global data_columns
    global model
    with open("columns.json", 'r') as f:
        data_columns = json.load(f)['data_columns']
        car_models = data_columns[10:]

    with open('Car Price Predictor model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Loading Saved artifacts...done")


if __name__ == '__main__':
    load_saved_artifacts()
    print(predict_price('Maruti Swift Dzire VDI', 145500, 1, 1, 1, 0, 23.40, 1248.0, 74.00, 5.0, 6))


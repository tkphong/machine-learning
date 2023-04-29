from fastapi import FastAPI
from forecast_temp import forecast_temp
from pydantic import BaseModel
from sklearn.ensemble import VotingClassifier
import pickle
import numpy as np
import datetime
import pytz
from db import requests_db
from forecast_temp import forecast_temp
from chatbot import chatbot

from fastapi.middleware.cors import CORSMiddleware

# Load model and stuff
loaded_knn = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/knn.sav', 'rb'))
loaded_rf = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/rf.sav', 'rb'))
loaded_svm = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/svm.sav', 'rb'))
loaded_dtree = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/dtree.sav', 'rb'))
loaded_gbdt = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/gbdt.sav', 'rb'))
loaded_scaler = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/scaler.sav', 'rb'))
loaded_voting = pickle.load(open('/home/phong/WeatherRecommend_MDP_ML/training_phase/voting.sav', 'rb'))
weather_cond = ['Clear', 'Cloud', 'Sunny', 'Rainy']
# End of model and stuff


class WeatherData(BaseModel):
    temperature: float
    wind_speed: float
    pressure: int
    humidity: int
    vis_km: int
    cloud: int


class ChatbotInput(BaseModel):
    input: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/v1/")
def predict(weather_data: WeatherData):
    X = np.array([[weather_data.temperature, weather_data.wind_speed, weather_data.pressure,
                 weather_data.humidity, weather_data.vis_km, weather_data.cloud]])
    X = loaded_scaler.transform(X)
    # tmp_arr = [0, 0, 0, 0]
    # nn1_arr = loaded_nn1.predict(X)
    # nn2_arr = loaded_nn2.predict(X)
    # dtree_arr = loaded_dtree.predict(X)
    # rf_arr = loaded_rf.predict(X)
    # svm_arr = loaded_svm.predict(X)
    # logreg_arr = loaded_logreg.predict(X)
    # nb_arr = loaded_nb.predict(X)
    #--------------------------------#
    voting_arr = loaded_voting.predict(X)
    counts = np.bincount(voting_arr)
    most_common = np.argmax(counts)
    #------------------------------#
    # ensample process
    # if nn1_arr == 0:
    #     tmp_arr[0] += 0.9
    # elif nn1_arr == 1:
    #     tmp_arr[1] += 0.8
    # elif nn1_arr == 2:
    #     tmp_arr[2] += 1
    # elif nn1_arr == 3:
    #     tmp_arr[3] += 0.9

    # if nn2_arr == 0:
    #     tmp_arr[0] += 0.9
    # elif nn2_arr == 1:
    #     tmp_arr[1] += 0.8
    # elif nn2_arr == 2:
    #     tmp_arr[2] += 1
    # elif nn2_arr == 3:
    #     tmp_arr[3] += 0.9

    # if dtree_arr == 0:
    #     tmp_arr[0] += 0.9
    # elif dtree_arr == 1:
    #     tmp_arr[1] += 0.8
    # elif dtree_arr == 2:
    #     tmp_arr[2] += 1
    # elif dtree_arr == 3:
    #     tmp_arr[3] += 0.9

    # if rf_arr == 0:
    #     tmp_arr[0] += 1
    # elif rf_arr == 1:
    #     tmp_arr[1] += 0.8
    # elif rf_arr == 2:
    #     tmp_arr[2] += 1.2
    # elif rf_arr == 3:
    #     tmp_arr[3] += 1

    # if svm_arr == 0:
    #     tmp_arr[0] += 0.7
    # elif svm_arr == 1:
    #     tmp_arr[1] += 0.5
    # elif svm_arr == 2:
    #     tmp_arr[2] += 0.8
    # elif svm_arr == 3:
    #     tmp_arr[3] += 0.7

    # if logreg_arr == 0:
    #     tmp_arr[0] += 0.7
    # elif logreg_arr == 1:
    #     tmp_arr[1] += 0.5
    # elif logreg_arr == 2:
    #     tmp_arr[2] += 0.8
    # elif logreg_arr == 3:
    #     tmp_arr[3] += 0.7

    # if nb_arr == 0:
    #     tmp_arr[0] += 0.6
    # elif nb_arr == 1:
    #     tmp_arr[1] += 0.5
    # elif nb_arr == 2:
    #     tmp_arr[2] += 0.7
    # elif nb_arr == 3:
    #     tmp_arr[3] += 0.6

    # if tmp_arr[0] == max(tmp_arr):
    #     res = 0
    # elif tmp_arr[1] == max(tmp_arr):
    #     res = 1
    # elif tmp_arr[2] == max(tmp_arr):
    #     res = 2
    # elif tmp_arr[3] == max(tmp_arr):
    #     res = 3


    return {'Condition': weather_cond[most_common]}


@app.get("/api/v2/{hours}")
def predict_temp(hours: int):
    temps = requests_db.get_temp()
    predict_temps = forecast_temp.forecast_temperature(temps)

    time_now_zone = datetime.datetime.now(pytz.timezone("Asia/Ho_Chi_Minh"))
    current_hour = int(time_now_zone.strftime('%H'))
    filter_predict_temps = predict_temps[current_hour+1:current_hour+1+hours]

    return [{"temperature": predict_temp} for predict_temp in filter_predict_temps]


@app.post("/api/v3/")
def chatbot_response(chatbot_input: ChatbotInput):
    return {"response": chatbot.bot_response(chatbot_input.input)}

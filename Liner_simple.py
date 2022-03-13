#ライブラリのインポート
# データ加工・処理・分析ライブラリ
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
import sklearn

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns



#データ分割（訓練データとテストデータ）のためのインポート
from sklearn.model_selection import train_test_split
# 単回帰のモデル構築のためのインポート
from sklearn.linear_model import LinearRegression
# ロジスティクス回帰モデルの構築のためのインポート
from sklearn.linear_model import LogisticRegression

#データを整形
weather = pd.read_csv('weather_info.csv' , engine='python')
weather_renew = weather[weather['Times of Day'] == 19]
weather_about = weather_renew[['date','pressure_state','temperature','Humidity','wind_speed']]
lot = pd.read_csv('result.csv' , engine='python')
lot_wed = pd.merge(weather_about, lot, on='date')
lot_wed_new = lot_wed[['pressure_state','temperature','Humidity','wind_speed','Winning_number']]
win_list = []
for i in range(len(lot_wed_new)):
    lot_wed_sub = lambda x: str(int(lot_wed_new.iloc[x,4])) if int(lot_wed_new.iloc[x,4]) >= 100 else str(int(lot_wed_new.iloc[x,4])).zfill(3)
    x = [int(a) for a in lot_wed_sub(i)]
    win_list.append(x)
    
win_df = pd.DataFrame(win_list,
                  columns=['win_first', 'win_second', 'win_third'])
lot_wed_df = lot_wed_new.join(win_df)
lot_wed_df = lot_wed_df[['pressure_state','temperature','Humidity','wind_speed','win_first', 'win_second', 'win_third']]
lot_wed_df.head()

class prac_Log:
    def __init__(salf, X_predict):
        X_predict_new = pd.DataFrame(X_predict).T
        X_predict_new.columns = ['pressure_state','temperature','Humidity','wind_speed']
        
        #1桁目
        #目的変数にwin_firstを指定、説明変数にそれ以外を指定
        X_f = lot_wed_df.drop(['win_first', 'win_second', 'win_third'], axis = 1)
        y_f = lot_wed_df['win_first']

        #訓練データとテストデータに分ける
        Xf_train,Xf_test,yf_train,yf_test = train_test_split(X_f,y_f,test_size = 0.5,random_state = 0)

         # ロジスティック回帰クラスの初期化と学習
        model_Logistic_f = LogisticRegression(max_iter=1500)
        model_Logistic_f.fit(Xf_train,yf_train)
        
        #2桁目
        #目的変数にwin_secondを指定、説明変数にそれ以外を指定
        X_s = lot_wed_df.drop(['win_first', 'win_second', 'win_third'], axis = 1)
        y_s = lot_wed_df['win_second']

        #訓練データとテストデータに分ける
        Xs_train,Xs_test,ys_train,ys_test = train_test_split(X_s,y_s,test_size = 0.5,random_state = 0)

         # ロジスティック回帰クラスの初期化と学習
        model_Logistic_s = LogisticRegression(max_iter=1500)
        model_Logistic_s.fit(Xs_train,ys_train)
        
        #3桁目
        #目的変数にwin_thirdを指定、説明変数にそれ以外を指定
        X_t = lot_wed_df.drop(['win_first', 'win_second', 'win_third'], axis = 1)
        y_t = lot_wed_df['win_third']

        #訓練データとテストデータに分ける
        Xt_train,Xt_test,yt_train,yt_test = train_test_split(X_t,y_t,test_size = 0.5,random_state = 0)

         # ロジスティック回帰クラスの初期化と学習
        model_Logistic_t = LogisticRegression(max_iter=1500)
        model_Logistic_t.fit(Xt_train,yt_train)
        
        print('予測される当選番号は%d%d%dです' % (model_Logistic_f.predict(X_predict_new),model_Logistic_s.predict(X_predict_new),model_Logistic_t.predict(X_predict_new)))
              
X_predict = [1017.9,12.3,43,2.6]
prac_log = prac_Log(X_predict)

from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor #using another algorithm called local outlier factor
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import plotly.graph_objects as go
import plotly


main = tkinter.Tk()
main.title("Catch Me If You Can: Detecting Pickpocket Suspects from Large-Scale Transit Records") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y, X_train, X_test, y_train, y_test, dataset, passenger, scaler, pid
accuracy = []
precision = []
recall = [] 
fscore = []

def uploadDataset(): #function to upload tweeter profile
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename, nrows=100)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset))

def processDataset():
    text.delete('1.0', END)
    global dataset, passenger, scaler, X, pid
    pid = int(user_list.get())
    dataset.drop(['pickup_datetime','dropoff_datetime','id','store_and_fwd_flag'], axis = 1,inplace=True)
    passenger = dataset.loc[dataset['passenger_id'] == pid]
    X = passenger.values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Dataset Preprocessing Completed for selected Passenger : "+str(pid)+"\n\n")
    text.insert(END,str(X))

#function to calculate all metrics
def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")

def runOCS():
    global accuracy, precision, recall, fscore, pid
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    global X, Y
    text.delete('1.0', END)
    ocs_model =  OneClassSVM()
    ocs_model.fit(X)
    Y = ocs_model.predict(X)
    Y[Y == 1] = 0
    Y[Y == -1] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    unique, count = np.unique(Y, return_counts = True)
    print(unique)
    print(count)
    pick_pocket = None
    if count[0] >= count[1]:
        pick_pocket = "Person "+str(pid)+" is not pick pocket suspected"
    else:
        pick_pocket = "Person "+str(pid)+" is pick pocket suspected"


    text.insert(END,pick_pocket+"\n\n")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    calculateMetrics("OCS with Decision Tree", y_test, predict)

    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("OCS with SVM", y_test, predict)

    lr = DecisionTreeClassifier()
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    calculateMetrics("OCS with LogisticRegression", y_test, predict)


def runTSSVM():
    global accuracy, precision, recall, fscore, pid
    global X, Y, passenger
    
    ts =  LocalOutlierFactor(novelty=True)
    ts.fit(X)
    Y = ts.predict(X)
    Y[Y == 1] = 0
    Y[Y == -1] = 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    calculateMetrics("TS-SVM with Decision Tree", y_test, predict)

    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("TS-SVM with SVM", y_test, predict)

    lr = DecisionTreeClassifier()
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    calculateMetrics("TS-SVM with LogisticRegression", y_test, predict)

    src_lat = passenger['pickup_latitude'].ravel()
    src_lon = passenger['pickup_longitude'].ravel()
    des_lat = passenger['dropoff_latitude'].ravel()
    des_lon = passenger['dropoff_longitude'].ravel()
    fig = go.Figure()
    for i in range(len(src_lat)):
        fig.add_trace(go.Scattergeo(lat = [src_lat[0],src_lon[0]],lon = [des_lat[0], des_lon[0]],mode = 'markers+lines',marker = {'size': 10}, line = dict(width = 4.5, color = 'blue')))
    fig.update_layout(title_text = 'Connection Map Depicting Flights from Brazil to All Other Countries', height=700, width=900, margin={"t":0,"b":0,"l":0, "r":0, "pad":0}, showlegend=False)
    plotly.offline.plot(fig)


def graph():
    df = pd.DataFrame([['Decision Tree with OCS','Precision',precision[0]],['Decision Tree with OCS','Recall',recall[0]],['Decision Tree with OCS','F1 Score',fscore[0]],['Decision Tree with OCS','Accuracy',accuracy[0]],
                       ['SVM with OCS','Precision',precision[1]],['SVM with OCS','Recall',recall[1]],['SVM with OCS','F1 Score',fscore[1]],['SVM with OCS','Accuracy',accuracy[1]],
                       ['Logistic Regression with OCS','Precision',precision[2]],['Logistic Regression with OCS','Recall',recall[2]],['Logistic Regression with OCS','F1 Score',fscore[2]],['Logistic Regression with OCS','Accuracy',accuracy[2]],
                       ['Decision Tree with TS-SVM','Precision',precision[3]],['Decision Tree with TS-SVM','Recall',recall[3]],['Decision Tree with TS-SVM','F1 Score',fscore[3]],['Decision Tree with TS-SVM','Accuracy',accuracy[3]],
                       ['SVM with TS-SVM','Precision',precision[4]],['SVM with TS-SVM','Recall',recall[4]],['SVM with TS-SVM','F1 Score',fscore[4]],['SVM with TS-SVM','Accuracy',accuracy[4]],
                       ['Logistic Regression with TS-SVM','Precision',precision[5]],['Logistic Regression with TS-SVM','Recall',recall[5]],['Logistic Regression with TS-SVM','F1 Score',fscore[5]],['Logistic Regression with TS-SVM','Accuracy',accuracy[5]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def exit():
    main.destroy()

    
font = ('times', 16, 'bold')
title = Label(main, text='Catch Me If You Can: Detecting Pickpocket Suspects from Large-Scale Transit Records')
title.config(bg='lightcyan1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Transit Records Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)

l1 = Label(main, text='Passenger ID:')
l1.config(font=font1)
l1.place(x=360,y=550)
mid = []
mid.append("1")
mid.append("2")
user_list = ttk.Combobox(main,values=mid,postcommand=lambda: user_list.configure(values=mid))
user_list.place(x=560,y=550)
user_list.current(0)
user_list.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=processDataset)
preprocessButton.place(x=50,y=600)
preprocessButton.config(font=font1) 

ocsButton = Button(main, text="Run One Class SVM", command=runOCS)
ocsButton.place(x=250,y=600)
ocsButton.config(font=font1) 

lofButton = Button(main, text="Run Propose Two-Step SVM (TS-SVM)", command=runTSSVM)
lofButton.place(x=460,y=600)
lofButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=800,y=600)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=1070,y=600)
exitButton.config(font=font1) 

main.config(bg='steel blue')
main.mainloop()

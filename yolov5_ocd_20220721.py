import torch
import os
print(torch.cuda.is_available())
print(os.getcwd())
#yolov5ディレクトリに移動
os.chdir('./yolov5')
print(os.getcwd())
#tourch hubを使って学習済みモデルをmodelに格納
import torch
#cocomodelを読み込む場合
model = torch.hub.load("../yolov5","yolov5s",source='local')
#ocd検出モデルを読み込む場合　xx.ptモデルをyolov5ディレクトリに置いておく
#model = torch.hub.load('','custom',path='ocd_20220413.pt',source='local',force_reload=True)
#torchhubモデルの検出時のパラメーター設定
model.conf=0.4
model.iou=0.45
model.multi_label=False
model.max_det=5

import PySimpleGUI as sg
#すべての警告の非表示
import warnings
warnings.simplefilter('ignore')

value = [0, 1, 2]#cameraの番号設定
threshould=0.4 #閾値の初期設定
#カメラ番号入力のポップアップ 外部入力なら1か2
cameranum = sg.popup_get_text("input camera number(0,1,2)", title="camera number", default_text="1")
cameranum =int(cameranum)
#GUIの初期設定
sg.theme('DarkBlue15')
layout = [
   [sg.Image(key='img1',), sg.Image(key='img2',)],
   #[[sg.Text('select camera'),sg.Listbox(value, size=(10, 3),key=('cameranum'))]],
   [[sg.Text('confidence thredshold'),sg.Slider(range=(0.1,1.0),default_value=0.4, resolution=0.1 ,orientation='h',
   size=(20,2),enable_events=True, key='slider',)]],
    [[sg.Button('Start', size=(10, 1)), sg.Button('Stop', size=(10,1))]],

    [sg.Button('Exit', size=(10, 1))],]
#記録イベントのためにstartedという関数を定義               
started= False
#webカメラで検出
import cv2
import time
import datetime as dt


#camera = cv2.VideoCapture('./data/images/ocd.mp4')
#pysimpleGUIのwindow定義
window = sg.Window("webカメラ画面", location=(100,100),layout=layout, size=(1000,1000),resizable=True,finalize=True)
event, values = window.read(timeout=20)
#cameranum = int(values['cameranum'])
#if event == ['caneranum']:
#    cameranum=int(values['cameranum'])
cap = cv2.VideoCapture(cameranum) #外部入力カメラを使用する場合0を1に変更
# ビデオ記録用の変数定義
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('result_{}.mp4'.format(dt.datetime.now()), fourcc, fps, (w, h))
while True:
    ret, imgs = cap.read()
    if not ret:
        break
    h,w,ch=imgs.shape
# キャプチャーした画像をpngに変換
    image1 = cv2.imencode('.png', imgs)[1].tobytes()                   
# Imageの内容を更新
    window['img1'].update(data=image1)#web cameraの画像を直接表示
    event, values = window.read(timeout=0)
    """if event == ['caneranum']:
        cameranum=int(values['cameranum'])
        cap=cv2.VideoCapture(cameranum)
        ret,img = cap.read()"""

    if values['slider']:
      threshould=values['slider']
      model.conf=threshould

    if event == 'Start':
        started = True
    if started == True:
        #画面を10*10分割して1/10のところから9/10のところまで切り抜き
        im_cropped=imgs[round(h/10):round(h/10)*9, round(w/10):round(w/10)*9,:]
        results = model(im_cropped)
        for *box, conf, cls in results.xyxy[0]:
            s = model.names[int(cls)]+":"+'{:.1f}'.format(float(conf)*100)
            cv2.rectangle(im_cropped,
                      (int(box[0]),int(box[1])),
                      (int(box[2]),int(box[3])),color=(0,255,0),thickness=4)
            cv2.putText(im_cropped, s, (int(box[0]),int(box[1])-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1.0,
                   color=(0,0,255),thickness =2)          
    # キャプチャーした画像をpngに変換
            image2 = cv2.imencode('.png', im_cropped)[1].tobytes()
    # Imageの内容を更新
            window['img2'].update(data=image2)       

    if event =='Stop':
        started = False
        out.release()

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

cap.release()    
window.close()
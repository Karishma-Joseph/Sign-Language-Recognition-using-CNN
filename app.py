from flask import Flask, render_template, Response, session, request, redirect, url_for
from camera import VideoCamera
from PIL import Image
import io
import numpy as np
import cv2, pickle
import os
import sqlite3
from keras.models import load_model
from threading import Thread
from time import time


app = Flask(__name__)


old_text = ""
word = ""
count_same_frame = 0
num_frames = 0
bg = None
thresholded = np.ones((50, 50))
aWeight = 0.5
top, right, bottom, left = 80, 350, 295, 590
x, y, w, h = 300, 100, 300, 300


def gen(camera):
    while True:
        frame = camera.get_frame()
        image = Image.open(io.BytesIO(frame)).convert('RGB')
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        recognize(open_cv_image)#have to finish this
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genT():
    while True:
        frame = thresholded
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

learnword = ""
def genL():
    global learnword
    while True:
        frames = [open('ASL/'+letter + '.jpg', 'rb').read() for letter in learnword]
        frame = frames[int(time()) % len(learnword)]
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/')
def index():
    return render_template('final.html')

@app.route('/Stream')
def stream():
    return render_template('livestream.html', word = word)

@app.route('/learn_feed')
def learn_feed():
    return Response(genL(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/Learn',methods = ['POST', 'GET'])
def learn():
    if request.method == 'POST':
        global learnword
        learnword = request.form['word']
    return render_template('learn.html')



def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape

image_x, image_y = 50, 50

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class        

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')
keras_predict(model, np.zeros((50, 50), dtype=np.uint8))

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]

def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    text = ""
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab*100 > 70:
        text = get_pred_text_from_db(pred_class)
    return text


def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

# Function - To segment the region of hand in the image
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return (thresholded, cnts)




def recognize(img):
    text = ""
    global count_same_frame
    global num_frames
    global old_text
    global word
    global thresholded
    #img = cv2.resize(img, (640, 480))

    # flip the frame so that it is not the mirror view
    #img = cv2.flip(img, 1)

    # clone the frame
    clone = img.copy()

    # get the height and width of the frame
    (height, width) = img.shape[:2]

    # get the ROI
    roi = img[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # to get the background, keep looking till a threshold is reached
    # so that our running average model gets calibrated
    if num_frames < 30:
        run_avg(gray, aWeight)
    else:
        # segment the hand region
        hand = segment(gray)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, contours) = hand
    
            if len(contours) > 0:
                contour = max(contours, key = cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    text = get_pred_from_contour(contour, thresholded)
                    if old_text == text:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0
                    if count_same_frame > 20:
                        word = word + text
                        count_same_frame = 0

                elif cv2.contourArea(contour) < 1000:
                    text = ""
                    word = ""
            else:
                text = ""
                word = ""
            old_text = text
    num_frames += 1

@app.route('/resetbackground',methods = ['POST'])
def resetbackground():
    if request.method == 'POST':
        global num_frames
        num_frames = 0
        return redirect(url_for('stream'))





@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thresh_feed')
def thresh_feed():
    return Response(genT(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/word_feed')
def word_feed():
    def generate():
        yield word  # return also will work
    return Response(generate(), mimetype='text') 



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
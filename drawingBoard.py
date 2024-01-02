import numpy as np
import tkinter as tk
import math
import os
import threading
from convolution import doubleConvGPU

firstArrSize = 784
firstArr = np.zeros(firstArrSize,dtype=np.float16)
secondArrSize = 256
secondArr = np.ndarray(secondArrSize,dtype=np.float16)
thirdArrSize = 128
thirdArr = np.ndarray(thirdArrSize,dtype=np.float16)
outputs = np.ndarray(10,dtype=np.float16)
correctValues = np.ndarray(10,dtype=np.int16)
np.set_printoptions(formatter={'float': lambda x: "{0:1.3f}".format(x)})

window = tk.Tk()

drawingFrame = tk.Frame(window,borderwidth=2,relief='solid')
drawingFrame.grid(row=0,column=0,padx=10,pady=15)
board = tk.Canvas(drawingFrame,height=196,width=196,highlightthickness=0,bg='white')
board.grid(row=0,column=0)
buttonFrame = tk.Frame(window,padx=10,pady=10)
buttonFrame.grid(row=0,column=1)

def setWAndB():
  global link12, link23, link34, biases2, biases3, biases4
  dir_path = os.path.dirname(os.path.realpath(__file__))
  file = open(dir_path + "/dataStore/link12.npy",'rb')
  link12 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/link23.npy",'rb')
  link23 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/link34.npy",'rb')
  link34 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/biases2.npy",'rb')
  biases2 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/biases3.npy",'rb')
  biases3 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/biases4.npy",'rb')
  biases4 = np.load(file)
  file.close()

def checkNum():
    global firstArr
    
    drawing = firstArr.reshape((28,28))
    for y in range(len(drawing)):
      for x in range(len(drawing[y])):
        if(drawing[y][x]==1):
          try:
            if drawing[y-1][x] == 0:
              drawing[y-1][x] = 0.5
          except:
              pass
          try:
            if drawing[y+1][x] == 0:
              drawing[y+1][x] = 0.5
          except:
              pass
          try:
            if drawing[y][x-1] == 0:
              drawing[y][x-1] = 0.5
          except:
              pass
          try:
            if drawing[y][x+1] == 0:
              drawing[y][x+1] = 0.5
          except:
              pass
    
    outputText = ""
    for row in drawing:
        for val in row:
            if(val == 0):
                outputText += "_"
            elif(val == 0.5):
               outputText += "+"
            else:
               outputText += "#" 
        outputText += "\n"
    print(outputText)

    firstArr = doubleConvGPU(np.array([drawing], dtype=np.float32), True)[0]
    secondArr = 1/(1 + np.exp(-((firstArr @ link12)+biases2)))
    thirdArr = 1/(1 + np.exp(-((secondArr @ link23)+biases3)))
    outputs = 1/(1 + np.exp(-((thirdArr @ link34)+biases4)))
    print('value:', np.argmax(outputs))
    print('outputs:', outputs)

mouseClicked = False

def move(event):
    global mouseClicked
    if mouseClicked:
        x, y = math.floor((event.x)/7), math.floor((event.y)/7)
        xPixel, yPixel = x*7, y*7
        firstArr[x+(28*y)] = 1
        event.widget.create_rectangle(xPixel,yPixel,xPixel+7,yPixel+7,fill='black')
        if(x > 0):
            firstArr[x-1+(28*y)] = 1
            event.widget.create_rectangle(xPixel-7,yPixel,xPixel,yPixel+7,fill='black')
        if(x < 27):
            firstArr[x+1+(28*y)] = 1
            event.widget.create_rectangle(xPixel+7,yPixel,xPixel+14,yPixel+7,fill='black')
        if(y > 0):
            firstArr[x+(28*(y-1))] = 1
            event.widget.create_rectangle(xPixel,yPixel-7,xPixel+7,yPixel,fill='black')
        if(y < 27):
            firstArr[x+(28*(y+1))] = 1
            event.widget.create_rectangle(xPixel,yPixel+7,xPixel+7,yPixel+14,fill='black')

def click(event):
    global mouseClicked
    mouseClicked = True
    move(event)

def release(event):
    global mouseClicked
    mouseClicked = False

def exit(event):
    global mouseClicked
    mouseClicked = False

board.bind('<Button-1>', click)
board.bind('<Motion>', move)
board.bind('<ButtonRelease-1>', release)
board.bind('<Leave>', exit)

def submitButtonClicked(event):
    global firstArr

    def enableEm():
        global firstArr
        thread.join()
        firstArr = np.zeros(firstArrSize,dtype=np.float16)
        button1.configure(state=tk.NORMAL)
        button2.configure(state=tk.NORMAL)

    thread = threading.Thread(target=checkNum)
    thread.start()
    button1.configure(state=tk.DISABLED)
    button2.configure(state=tk.DISABLED)
    board.delete("all")
    thread2 = threading.Thread(target=enableEm)
    thread2.start()

def resetButtonClicked(event):
    global firstArr
    board.delete("all")
    firstArr = np.zeros(firstArrSize,dtype=np.float16)

button1 = tk.Button(buttonFrame, text='submit',font=('Helvetica', '12'), highlightthickness=0,borderwidth=2,compound=tk.CENTER)
button1.grid(row=0,column=0)
button1.bind("<Button-1>", submitButtonClicked)

button2 = tk.Button(buttonFrame, text='reset',font=('Helvetica', '12'),highlightthickness=0,borderwidth=2,compound=tk.CENTER)
button2.grid(row=1,column=0)
button2.bind("<Button-1>", resetButtonClicked)

if(__name__ == "__main__"):
  setWAndB()
  window.mainloop()
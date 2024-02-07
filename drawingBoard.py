import numpy as np
import tkinter as tk
import math
import os
import threading
from convolution import convGPU
from skimage.measure import block_reduce
from initWeightsAndBiases import inputArrSize, hiddenL1ArrSize, hiddenL2ArrSize
import matplotlib.pyplot as plt

BOARD_SIZE = 196
drawing = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
window = tk.Tk()
drawingFrame = tk.Frame(window,borderwidth=2,relief='solid')
drawingFrame.grid(row=0,column=0,padx=10,pady=15)
board = tk.Canvas(drawingFrame, height=BOARD_SIZE, width=BOARD_SIZE, highlightthickness=0, bg='white')
board.grid(row=0,column=0)
buttonFrame = tk.Frame(window,padx=10,pady=10)
buttonFrame.grid(row=0,column=1)
np.set_printoptions(formatter={'float': lambda x: "{0:1.3f}".format(x)})

def setWAndB():
  global link12, link23, link34, biases2, biases3, biases4, kernels1, kernels2, kernels1Biases, kernels2Biases
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
  file = open(dir_path + "/dataStore/kernels1.npy",'rb')
  kernels1 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/kernels2.npy",'rb')
  kernels2 = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/kernels1Biases.npy",'rb')
  kernels1Biases = np.load(file)
  file.close()
  file = open(dir_path + "/dataStore/kernels2Biases.npy",'rb')
  kernels2Biases = np.load(file)
  file.close()

def checkNum():
    reducedDrawing = block_reduce(drawing, block_size=7, func=np.mean)
    for row in reducedDrawing:
        for val in row:
            print("#" if round(val) ==1 else "_", end=" ")
        print()
    #plt.imshow(reducedDrawing)
    #plt.show()
    (filteredImgs, matricesForPassedPixels, smallImages) = convGPU(reducedDrawing.reshape((1,28,28)), kernels1, kernels1Biases)
    (filteredImgs2, matricesForPassedPixels2, smallImages2) = convGPU(smallImages, kernels2, kernels2Biases)

    firstArr = smallImages2.reshape((1, inputArrSize))
    hiddenL1Arr = ((firstArr @ link12) + biases2)/inputArrSize
    hiddenL1Arr = np.where(hiddenL1Arr<0,0.01*hiddenL1Arr,hiddenL1Arr)
    hiddenL2Arr = ((hiddenL1Arr @ link23) + biases3)/hiddenL1ArrSize
    hiddenL2Arr = np.where(hiddenL2Arr<0,0.01*hiddenL2Arr,hiddenL2Arr)
    outputs = ((hiddenL2Arr @ link34) + biases4)/hiddenL2ArrSize
    outputs = np.where(outputs<0,0.01*outputs,outputs)

    print('value:', np.argmax(outputs))
    print('outputs:', outputs)

mouseClicked = False

def move(event):
    global drawing
    if mouseClicked:
        drawing[event.y-7:event.y+7, event.x-7:event.x+7] = 1
        event.widget.create_rectangle(event.x-7, event.y-7, event.x+7, event.y+7, fill='black')

def click(event):
    global mouseClicked
    mouseClicked = True
    move(event)

def release(event):
    global mouseClicked
    mouseClicked = False

board.bind('<Button-1>', click)
board.bind('<Motion>', move)
board.bind('<ButtonRelease-1>', release)

def submitButtonClicked(event):
    global drawing

    def enableEm():
        global drawing
        thread.join()
        drawing *= 0
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
    global drawing
    drawing *= 0
    board.delete("all")

button1 = tk.Button(buttonFrame, text='submit',font=('Helvetica', '12'), highlightthickness=0,borderwidth=2,compound=tk.CENTER)
button1.grid(row=0,column=0)
button1.bind("<Button-1>", submitButtonClicked)

button2 = tk.Button(buttonFrame, text='reset',font=('Helvetica', '12'),highlightthickness=0,borderwidth=2,compound=tk.CENTER)
button2.grid(row=1,column=0)
button2.bind("<Button-1>", resetButtonClicked)

if(__name__ == "__main__"):
  setWAndB()
  window.mainloop()
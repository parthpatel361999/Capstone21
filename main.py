import PIL
from PIL import Image,ImageTk
import pytesseract
import cv2
import cleaningObj
import pickle

from tkinter import *
width, height = 900, 900
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.geometry('750x540')
root.title("PROJECT TITLE")
root.bind('<Escape>', lambda e: root.quit())
def sclick():
    filename = input("Enter file name")
    savefile = open(filename, 'wb')
    pickle.dump(lol, savefile)
    print(lol.rightFace)

def lclick():
    filename2 = input("Enter file name")
    readfile = open(filename2, 'rb')
    loadobject = pickle.load(readfile)
    cleaningObj.loadupdate(loadobject)
    print(loadobject.rightFace)

#lmain = Label(root,height=480, width=640)
sbutton = Button(root, text="SAVE", height=2, width=5, command=sclick)
sbutton.grid(column=1, row=0)
lbutton = Button(root, text="LOAD", height=2, width=5, command=lclick)
lbutton.grid(column=0, row=0)
#lmain.grid(column=2, row=1)

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

lol = cleaningObj.CleaningObject([2,6,4],[7,6,2,3,6,4,6,4],[447,345],[37,37],[36,4,6])
#show_frame()
root.mainloop()

import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model


model = load_model("braintumor10Epoccategorical.h5")


root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("420x600")

label_result = Label(root, text="", font=("Arial", 16))
label_result.pack(pady=10)

panel = Label(root) 
panel.pack(pady=10)

current_prediction = None 


def read_image_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def select_image():
    global current_prediction
    path = filedialog.askopenfilename()
    if path:
       
        try:
            img = Image.open(path)
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            panel.configure(image=img_tk)
            panel.image = img_tk
        except Exception:
            label_result.config(text="❌ Cannot open image", fg="red")
            return

       
        image = read_image_unicode(path)
        if image is None:
            label_result.config(text="❌ Failed to read image", fg="red")
            return

      
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image) / 255.0
        input_img = np.expand_dims(image, axis=0)

       
        prediction = model.predict(input_img)
        class_id = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        current_prediction = (class_id, confidence)
        perc = int(round(confidence * 100))

      
        if class_id == 0:
            label_result.config(text=f"✅ No Tumor Detected – {perc}%", fg="green")
        else:
            label_result.config(text=f"⚠️ Tumor Detected – {perc}%", fg="red")

def delete_image():
   
    global current_prediction
    panel.configure(image=None)
    panel.image = None
    label_result.config(text="", fg="black")
    current_prediction = None

def clear_screen():
   
    delete_image()
    label_result.config(text="Select an image…", fg="black")


btn_frame = tk.Frame(root)
btn_frame.pack(side="bottom", pady=30)

btn_select = tk.Button(btn_frame, text="📁 Choose", command=select_image, font=("Arial", 12), width=12)
btn_select.grid(row=0, column=0, padx=6)

btn_delete = tk.Button(btn_frame, text="🗑 Delete", command=delete_image, font=("Arial", 12), width=12)
btn_delete.grid(row=0, column=1, padx=6)

btn_clear = tk.Button(btn_frame, text="🧹 Clear", command=clear_screen, font=("Arial", 12), width=12)
btn_clear.grid(row=0, column=2, padx=6)

root.mainloop()

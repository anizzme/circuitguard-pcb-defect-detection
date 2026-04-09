%%writefile pcb_backend.py

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper"
]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=6
    )

    model.load_state_dict(
        torch.load(
            "models/pcb_defect_model.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# DEFECT DETECTION
# -----------------------------
def detect_defects(template, test):

    test = cv2.resize(test,(template.shape[1],template.shape[0]))

    template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(template_gray,test_gray)
    diff = cv2.GaussianBlur(diff,(5,5),0)

    _,thresh = cv2.threshold(diff,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5,5),np.uint8)

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
    thresh = cv2.dilate(thresh,kernel,iterations=2)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)

    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 200 or area > 10000:
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        boxes.append((x,y,w,h))

    return boxes

# ------------------------------
# CLASSIFICATION
# ------------------------------
def classify_roi(roi):

    img = cv2.resize(roi,(128,128))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img = transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs,dim=1)
        conf,pred = torch.max(probs,1)

    return class_names[pred.item()], conf.item()

# ------------------------------
# COMPLETE PIPELINE
# ------------------------------
@st.cache_data
def process_pcb(template_path,test_path):

    template = cv2.imread(template_path)
    test = cv2.imread(test_path)

    boxes = detect_defects(template,test)

    output = test.copy()
    results = []

    for (x,y,w,h) in boxes:

        pad = 10
        x1 = max(0,x-pad)
        y1 = max(0,y-pad)
        x2 = min(test.shape[1],x+w+pad)
        y2 = min(test.shape[0],y+h+pad)

        roi = test[y1:y2,x1:x2]

        label,conf = classify_roi(roi)

        if conf < 0.80:
            continue

        #  HYBRID LOGIC
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        aspect_ratio = w / float(h + 1e-5)

        if label in ["Open_circuit", "Short"]:

            if edge_density < 0.08:
                label = "Open_circuit"

            elif edge_density > 0.15:
                label = "Short"

            else:
                if 0.8 < aspect_ratio < 1.2:
                    label = "Short"
                else:
                    label = "Open_circuit"

        results.append((label,conf))

        #  RED COLOR (BGR = (0,0,255))
        cv2.rectangle(output,(x,y),(x+w,y+h),(0,0,255),2)

        cv2.putText(
            output,
            f"{label} {conf:.2f}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    return output,results

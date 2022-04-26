from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
#import matplotlib.pyplot as plt
import os
from PIL import Image


IMAGE_SIZE = 227
MODEL_NAME = 'final_model.pt'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, 5, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout2d(0.2)
        self.fc4 = nn.Linear(256*8*8, 2048)
        self.fc5 = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.bn0(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 3, stride=2)   

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 3, stride=2)   
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.dropout1(x)
        
        x = x.view(-1, 256*8*8)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = x
        return output

model1 = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
# st.text(model1)

# test_transforms = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
# ])

st.header('PNEUMONIA DETECTOR')

st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader('Upload X-ray', type=['png','jpeg'])
if uploaded_file is not None:
    display_image = Image.open(uploaded_file)
    display_image = display_image.resize((IMAGE_SIZE,IMAGE_SIZE), resample=0) 
    st.image(display_image)






# # with st.echo(code_location='below'):

# st.header('STORM PREDICTOR')


# st.text('Chance of no sugar: ' + str(no_sugar)+'%')
# st.text('Chance of typical sugar: ' + str(typical_sugar)+'%')
# st.text('Chance of high sugar: ' + str(high_sugar)+'%')


#     st.subheader('FINAL EXPECTED PAYOUT VALUE: $'+str(final_expected_value))
#     st.subheader('DECISION: '+str(decision))
# else:
#     st.subheader('Chances of sugar should sum to 100')

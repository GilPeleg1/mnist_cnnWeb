import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn  # PyTorch neural network module
import torch.nn.functional as F  # PyTorch functional module

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # First convolutional layer (input channels: 1, output channels: 32, kernel size: 3x3, stride: 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Second convolutional layer (input channels: 32, output channels: 64, kernel size: 3x3, stride: 1)
        self.dropout1 = nn.Dropout(0.25)  # First dropout layer with 25% drop probability
        self.dropout2 = nn.Dropout(0.5)  # Second dropout layer with 50% drop probability
        self.fc1 = nn.Linear(9216, 128)  # First fully connected layer (input features: 9216, output features: 128)
        self.fc2 = nn.Linear(128, 10)  # Second fully connected layer (input features: 128, output features: 10)

    def forward(self, x):
        x = self.conv1(x)  # Apply first convolutional layer
        x = F.relu(x)  # Apply ReLU activation
        x = self.conv2(x)  # Apply second convolutional layer
        x = F.relu(x)  # Apply ReLU activation
        x = F.max_pool2d(x, 2)  # Apply max pooling with 2x2 kernel
        x = self.dropout1(x)  # Apply first dropout layer
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc1(x)  # Apply first fully connected layer
        x = F.relu(x)  # Apply ReLU activation
        x = self.dropout2(x)  # Apply second dropout layer
        x = self.fc2(x)  # Apply second fully connected layer
        output = F.log_softmax(x, dim=1)  # Apply log softmax activation
        return output  # Return the output
# Define your model architecture
model = Net()

# Load the saved parameters into the model
model.load_state_dict(torch.load("mnist_cnn.pt"))

# Set the model to evaluation mode
model.eval()

# Function to preprocess image

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize image to 28x28
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.ToTensor(),   # Convert image to tensor
        transforms.Normalize(mean=[0.1307], std=[0.3081])  # Normalize image
    ])
    image = preprocess(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to predict class
def predict_class(image):
    # Preprocess image
    preprocessed_img = preprocess_image(image)
    # Make prediction
    with torch.no_grad():
        predictions = model(preprocessed_img)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
    return "המספר בתמונה הוא: "+str(np.argwhere(probabilities.numpy()[0]== np.max(probabilities.numpy()[0]))[0,0])#np.argwhere(probabilities.numpy()==max(probabilities.numpy()))

# Streamlit UI
st.title('אתר לזיהוי תמונות בעזרת רשת קונבולוציה')
st.text('כנסו לתוכנת הצייר, ציירו מספר בין 0 ל 9 בלבן על רקע שחור, ותבדקו אם המודל שלנו עובד')
st.text("המודל עובד על תמונות בגודל 28 על 28 פיקסלים, ולכן התמונות עוברות שינוי גודל לפני")
st.text("שהן נכנסות לרשת, מה שיכול לשנות את הפרידקציה, ולכן עדיף לצייר מראש על קנבס ")
st.text("בגודל 28 על 28 פיקסלים")
uploaded_image = st.file_uploader("...בחר תמונה", type=['jpg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict_class(image)
    st.title("פרדיקציה: "+ prediction)
st.markdown('##')
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.image('cshev.png',caption='תיאור הליך הקלסיפיקציה של התמונה')

from flask import Flask, render_template, request
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.cnn_model import CNN  # Import your CNN model
import pandas as pd

app = Flask(__name__)

# Load your trained CNN model
cnn_model = CNN(num_classes=39)  # Adjust number of classes as needed
cnn_model.load_state_dict(torch.load('cnn_model.pth'))  # Load the trained model
cnn_model.eval()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(file)
        # Preprocess the image
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        # Predict using the model
        with torch.no_grad():
            output = cnn_model(image)
            _, predicted = torch.max(output, 1)
        # Return the prediction result
        return render_template('result.html', prediction=predicted.item())

if __name__ == '__main__':
    app.run(debug=True)

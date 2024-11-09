from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained('umutbozdag/plant-identity', num_labels=10, ignore_mismatched_sizes=True)

model_weights_path = 'model.pth'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),        
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']

    try:
        img = Image.open(file).convert('RGB')

        img_t = transform(img).unsqueeze(0).to(device)
    
        with torch.no_grad():
            outputs = model(img_t).logits
            _, predicted = torch.max(outputs, 1)

        class_names = ["Aloe Vera", "Areca Palm", "Boston Fern", "Chinese evergreen", "Dracaena", "Money Tree", "Peace lily", "Rubber Plant", "Snake Plant", "ZZ Plant"]
        predicted_class = class_names[predicted.item()]

        # return render_template('index.html', result=predicted_class, image = img)
        return jsonify(predicted_class)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
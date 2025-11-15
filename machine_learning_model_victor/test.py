import torch
import os
import torchvision.transforms as transforms
from PIL import Image
from cnn import CNN

"""
This script loads data from saved contours of a reciept and classifies then loads the CNN model and classifies the character of each one

Currently it works very poorly and is misclassifying most of the characters on my test receipt, this is due to the data the model was trained on.
The training dataset includes upper and lowercase letters for training but receipts tend to have mostly uppercase so I am going to try and find more datasets with 
more uppercase letters to bias the model towards classifying those
"""
def main():
        
    image_folder = "chars_rgb"
    images = [f for f in os.listdir(image_folder)]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),        # Ensure images are 32x32
        transforms.ToTensor(),              # Convert to tensor [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = CNN().to(device)

    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.eval()

    print("Model Loaded")

    output = []
    classes = ['#', '$', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    for image in images:
        # Load image
        img_path = os.path.join(image_folder, image)
        img = Image.open(img_path).convert("RGB")  # ensure RGB

        print(img)

        # Preprocess
        input_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

        # Forward pass
        with torch.no_grad():
            out = model(input_tensor)
            _, predicted = torch.max(out, 1)  # get class index

        print(f"{image}: Predicted class {classes[predicted.item()]}")
        output.append(classes[predicted.item()])

if __name__ == "__main__":
    main()

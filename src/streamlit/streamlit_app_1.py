import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

# Function to load Keras model
def load_keras_model(file_path):
    return tf.keras.models.load_model(file_path)

# Function to load PyTorch model
def load_pytorch_model(file_path):
    model = CustomResNet50(num_classes=38)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

# Load class indices from a JSON file
def load_class_indices(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Function to preprocess the uploaded image
def preprocess_image(image, model):
    """
    Preprocess an image to match the input size of the given model.

    Parameters:
        image (PIL.Image): The input image.
        model: The trained model (Keras or PyTorch).

    Returns:
        np.ndarray: The preprocessed image array for Keras.
        torch.Tensor: The preprocessed image tensor for PyTorch.
    """
    if isinstance(model, tf.keras.Model):  # For Keras models
        input_shape = model.input_shape[1:3]  # Extract input size from the model
        resized_image = image.resize(input_shape)
        image_array = np.array(resized_image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array
    elif isinstance(model, torch.nn.Module):  # For PyTorch models
        input_size = 256  # Default input size for many PyTorch models
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),  # Resize to input size
            transforms.ToTensor(),
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError("Unsupported model type. Please provide a Keras or PyTorch model.")

# Custom ResNet50 class to define our TL architecture
class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()

        # Initialize the base model without pre-trained weights
        self.base_model = models.resnet50(pretrained=False)

        
        # Modify the final fully connected layer to match the pre-trained model from the .pth file
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, 120)
      
        # Initialize the base model and load custom weights
        state_dict = torch.load('src/models/model_src/ResNet50-Plant-model-80.pth', map_location=torch.device('cpu'))
        self.base_model.load_state_dict(state_dict)
      
        # Modify the classifier
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the original FC layer
      
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)  # Return raw logits
        return x


# Grad-CAM function for Keras
def generate_grad_cam_keras(model, image, layer_name=None):
    # Automatically detect the input size of the model
    input_shape = model.input_shape[1:3]  # Get height and width of the input layer

    # Resize the input image to match the model's input size
    resized_image = image.resize(input_shape)
    img_array = img_to_array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # If layer_name is not provided, find the last convolutional layer
    if not layer_name:
        layer_name = next(
            (layer.name for layer in model.layers[::-1] if isinstance(layer, tf.keras.layers.Conv2D)),
            None
        )
        if not layer_name:
            raise ValueError("No convolutional layer found in the model.")

    # Create a model that maps inputs to activations of the target layer and model output
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    # Compute the gradient of the loss with respect to the feature map
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Compute the Grad-CAM heatmap
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if max_value > 0:
        heatmap /= max_value
    heatmap = heatmap.numpy()

    # Resize the heatmap to the original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)

    # Overlay the heatmap on the image
    heatmap = np.array(heatmap)
    colormap = plt.cm.jet(heatmap / 255.0)[:, :, :3]  # Apply colormap
    overlay = (colormap * 255).astype(np.uint8)
    overlay_image = Image.blend(image.convert("RGBA"), Image.fromarray(overlay).convert("RGBA"), alpha=0.5)

    return overlay_image

# Grad-CAM function for PyTorch
def generate_grad_cam_pytorch(model, image, target_layer_name=None):
    """
    Generate Grad-CAM visualization for a PyTorch model.
  
    Parameters:
        model: PyTorch model
        image: PIL Image (input image)
        layer_name: str (name of the target convolutional layer), if not provided the function iterates through the
                    model's layer in reverse and finds the last Conv2d layer

    Returns:
        PIL Image with the Grad-CAM overlay
    """
    from torch.nn import Conv2d

    # Automatically detect the last convolutional layer if target_layer_name is not provided
    if target_layer_name is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, Conv2d):
                target_layer_name = name
                break
        if target_layer_name is None:
            raise ValueError("No convolutional layer found in the model.")

    # Transform and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the input size of the model
        transforms.ToTensor(),
    ])
    torch_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Hook to capture gradients and activations
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    # Register hooks on the target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(save_activations)
    target_layer.register_backward_hook(save_gradients)

    # Forward pass
    model.eval()
    output = model(torch_image)
    class_index = output.argmax(dim=1).item()

    # Backward pass for the target class
    model.zero_grad()
    loss = output[0, class_index]
    loss.backward()

    # Compute Grad-CAM
    gradients = gradients[0].detach().cpu().numpy()
    activations = activations[0].detach().cpu().numpy()
    weights = np.mean(gradients, axis=(2, 3))  # Global average pooling of gradients
    grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)
    for i, w in enumerate(weights[0]):
        grad_cam += w * activations[0, i, :, :]

    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = grad_cam / grad_cam.max() if grad_cam.max() != 0 else grad_cam

    # Resize heatmap to match original image size
    grad_cam = np.uint8(255 * grad_cam)
    heatmap = Image.fromarray(grad_cam).resize(image.size, resample=Image.BILINEAR)

    # Overlay heatmap on the image
    heatmap = np.array(heatmap)
    colormap = plt.cm.jet(heatmap / 255.0)[:, :, :3]  # Apply colormap
    overlay = (colormap * 255).astype(np.uint8)
    overlay_image = Image.blend(image.convert("RGBA"), Image.fromarray(overlay).convert("RGBA"), alpha=0.5)

    return overlay_image


# Load the CSV file into a DataFrame
df = pd.read_csv("src/streamlit/plant_dataset.csv")

# start with the streamlit app

# sidebar and page navigation
st.sidebar.title("Table of contents")
pages = ["Home", "Data overview", "Data exploration", "Model: CNN","Model: Transfer Learning", "Model Interpretability", "Conclusion", "Predict your plant"]
page = st.sidebar.radio("Go to", pages)

# set a dynamic title for each page
# st.title(f"Planting Parents: {page}")

if page == pages[0]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("src/visualization/Planting_parents_logo.png", use_container_width=True)
        st.header("Introduction")
        st.write("Welcome to the site of the Planting Parents, where we have trained an AI model to recognise 14 different species of plants and whether they are sick or healthy by analysing 20 plant diseases."
             " We're a group of young parents who value life and share an interest in growing plant life as well. It has been a great and rewarding challenge to present to you this app with our findings and results." 
             " We sincerely hope you enjoy our page and that you may find it informative and recognise the plants you want to recognise. ")
        st.write("**The planting parents,**")
        st.write("**Lara, Ji-Young, Yannik & Niels**")


####################
# DATA OVERVIEW    #
####################


elif page == pages[1]:
    st.write("### Data overview")
    
    tab1, tab2, tab3 = st.tabs(["The numbers", "plants & diseases", "Pre-processing steps"])
    
    with tab1: # The numbers
        st.write("For our initial research into data, we stumbled upon many datasets that had some sort of plant recognition dataset of many different species from the website Kaggle. ")
        
        st.write("""
            We work with a database called ‘New Plant Diseases Dataset’ which is a public repository available on Kaggle.

    It includes multiple subfolders; Train, Valid and Test. These subfolders contain subfolders per plant where the target image files (jpg) are stored, totalling 87,900 images.
    All images have 256 x 256 as their image size:
                 
            Files in train_path:            70,295
        Files in valid_path:            17,572
        Files in test_path:                 33
        Total files:                    87,900
                 """)
        
        st.write("In the below table overview you'll find all the names of the plant species and plant diseases: ")
        st.image("src/visualization/Data_Exploration/Plants_&_Diseases_count.png", use_container_width=True)
    
    with tab2: # Plants & diseases
        st.write("")
        st.image("src/visualization/Data_Exploration/diseased_plant_samples.png")

        st.write("")
        st.image("src/visualization/Data_Exploration/healthy_plant_samples.png")

    with tab3: # Pre-processing steps
        st.write("As for the preprocessing phase we followed these steps to get to the analysis and visualisation phase of the files:")
        
        st.write("")

        st.write("""
        - Import kaggle hub and download dataset.
        - Define the paths.
        - Importing necessary libraries for data processing and visualization.
        - Creating a DataFrame with Train and Valid.
        - Check for missing values and duplicates between train and valid subset.
                 """)

        st.write("")

        st.write("The data was already well sorted, indexed and labelled. We only had to create an import path and created lists to work of from:")
        
        st.write("""
        - image_paths = []
        - species_labels = []
        - disease_labels = []
        - dataset_split = []
        """)

        st.write("We looked for missing values and duplicates even cross checked between the train and valid folders and found none. ")


####################
# DATA EXPLORATION #
####################


elif page == pages[2]:
    
    st.write("#### Data Exploration")

    tab1, tab2, tab3, tab4 = st.tabs(["General distribution", "Train vs validation ditribution", "Confusion matrix", "Data Transformation"])


    with tab1:
        st.write('''
        The following graphs give us an idea of the distribution of our dataset regarding the different plant species available and their diseases.
        ''')
        st.markdown("- Species")
        st.image("src/visualization/Data_Exploration/DExp_species_distribution.png")

        st.markdown("- Diseases")
        st.image("src/visualization/Data_Exploration/DExp_diseases_distribution.png")
    
        st.markdown("""<span style='color: red;'>The data shows a reasonable inbalance with the tomato class, but the number of total images for the other categories is high enough to use all of them for the modelling.</span>""", unsafe_allow_html=True)

    with tab2:
        st.write('''
        In here, we take a deeper insight of the distribution of our classes comparing the training dataset versus the valid dataset and their overall proportions.
        ''')
        st.markdown("- Species")
        st.image("src/visualization/Data_Exploration/Dexp_TrainValid_species.png")

        st.markdown("- Diseases")
        st.image("src/visualization/Data_Exploration/DExp_TrainValid_diseases.png")

    with tab3:
        st.write("The confusion matrix between species and diseases reveals the variability in the data and helps identify if a species is overrepresented or underrepresented.")
        st.image("src/visualization/Data_Exploration/DExp_CM_species_diseases.png")
        st.markdown("""<span style='color: red;'>We clearly observe that some species do not contain any disease examples and some species do not contain healthy examples.</span>""", unsafe_allow_html=True)


    with tab4:
        st.write('''
        To optimize the dataset, data transformations can be applied before training. This can help improve the model by, for example, emphasizing important features or removing unnecessary details, which can also save computational resources.
        Here we have filtered the images to remove background or detect edges.
        ''')
        st.markdown("1. Thresholding")
        st.image("src/visualization/Data_Exploration/DExp_thresholding_filter.png")

        st.markdown("2. Canny filtering")
        st.image("src/visualization/Data_Exploration/DExp_Canny_filter.png")

####################
# MODEL TRAINING   #
####################


elif page == pages[3]:
    st.write("### Model: CNN")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["CNN", "Dataset size", "Image size", "Learning rate", "Augmentation", "CNN layer"])


    with tab1: # Baseline CNN's (Ji)
        st.write("### Convolutional Neural Networks (CNNs) ")
        st.write('''
        In this section, we investigate the performance of Convolutional Neural Networks (CNNs), 
        a fundamental model choice for image analysis tasks 
        because of its ability to efficiently identify and interpret patterns and structures 
        at multiple levels of detail within visual data.
        ''')
        st.write('''
        To optimize the performance of our CNN model, we evaluated several parameters \
        that influence accuracy, generalization, and computational efficiency. 
        These parameters include dataset size, image resolution, learning rate, 
        data augmentation techniques, and the depth of the CNN architecture 
        (number of convolutional layers). By varying these parameters, 
        we aimed to identify the configurations that maximize model performance 
        for our image classification task.
        ''')
            

        st.write("### CNN Model Architecture ")
        st.write('''
        1. Convolutional Layers:
        These layers apply filters to the input data to extract features such as edges, textures, and other patterns. They help the network detect spatial features from images at different levels of abstraction.

        2. Batch Normalization Layers:
        These layers normalize the inputs of each layer to have a mean of 0 and a variance of 1. This speeds up training, stabilizes the learning process, and helps reduce the risk of overfitting.

        3. Max-Pooling Layers:
        These layers reduce the spatial dimensions of the feature maps by taking the maximum value from a pooling window (e.g., 2x2). This makes the model computationally efficient and helps it focus on the most important features while reducing sensitivity to spatial translations.

        4. Fully Connected Layers:
        These layers flatten the feature maps into a 1D vector and integrate the extracted features to perform the final classification or regression tasks. They are responsible for the decision-making process in the network.
                  ''')
        st.write('''

                 ''')
        st.write("### CNN Model: 2-layers (CNN-2x) vs 3-layers (CNN-3x) ")
        # Option 1
        col1, col2 = st.columns(2)
        with col1:
            st.image("src/visualization/CNN/Model-architecture_CNN-2.png", use_container_width=True)
        with col2:
            st.image("src/visualization/CNN/Model-architecture_CNN-3.png", use_container_width=True)
        # Option 2
        # st.image("Model-architecture_CNN-2.png", use_container_width=True)
        # st.image("Model-architecture_CNN-3.png", use_container_width=True)

        st.write("### Test parameters ")
        st.markdown('''
        **1. Size of the train dataset:** 20k vs 70k     

        **2. Image size:** 224x224 vs 256x256      

        **3. Learning rate:** 1e-3 ~ 1e-5     

        **4. Data augmentation**        

        **5. Number of CNN layers:** 2 layers vs 3 layers
        ''')

        st.write("### Performance Metrics ")
        st.markdown('''
        To monitor the performance of the models, the following evaluation metrics were used:

        **1. Training/Validation accuracy and loss:** measure learning progression
                    
        **2. Confusion matrix:** analyze classification performance across all classes
                    
        **3. Test dataset accuracy:** assess the models ability to generalize to unseen data.
        ''')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("src/visualization/CNN/ex_plot.png", use_container_width=True)
        with col2:
            st.image("src/visualization/CNN/ex_cm.png", use_container_width=True)
        with col3:
            st.image("src/visualization/CNN/TestDataset_33.png", use_container_width=True)

    with tab2:# Dataset size
        st.write("In this section, we tested different dataset size")
        st.image("src/visualization/CNN/1_CNN_Dataset_table.png", use_container_width=True)
        st.image("src/visualization/CNN/1_CNN_Dataset_graph+cm.png", use_container_width=True)

        st.markdown('''
        **Summary**
        - 70k dataset:
            - Improved training and validation accuracy, reducing validation loss.
            - Stronger diagonal dominance in confusion matrix, reflecting better classification performance with fewer misclassification.
        - **A larger dataset (70k) significantly boosts the model's performance compared to a smaller dataset (20k).**
        ''')
    with tab3: # Image size
        st.write("In this section, we tested different image sizes.")
        st.image("src/visualization/CNN/2_CNN_Image-size_table.png", use_container_width=True)
        st.image("src/visualization/CNN/2_CNN_Image-size_graph.png", use_container_width=True)

        with st.expander("Confusion matrix"):
            st.image("src/visualization/CNN/2_CNN_Image-size_cm.png", use_container_width=True)
    
        st.markdown('''
        **Summary**
        - The smoother loss curve and gradual accuracy improvement for larger image sizes (256x256) at lr = 1e-3 suggest better generalization and consistent learning.
        - Increasing the image size from 224x224 to 256x256 improves the test accuracy for both learning rates.
        ''')
    with tab4: # Learning rate
        st.write("In this section, we tested different learning rates.")
        st.image("src/visualization/CNN/3_CNN_Learningrate_table.png", use_container_width=True)
        st.image("src/visualization/CNN/3_CNN_Learningrate_graph.png", use_container_width=True)

        with st.expander("Confusion matrix"):
            st.image("src/visualization/CNN/3_CNN_Learningrate_cm.png", use_container_width=True)

        st.markdown('''
        **Summary**
        - For both architectures, a learning rate of 1e-5 (Models C and E) results in the best performance.
        - Lowering the learning rate significantly improves classification accuracy by enhancing diagonal dominance in the confusion matrix, and deeper architectures (CNN-3x) further amplify this improvement.
        - **Lower learning rates (1e-5) combined with deeper architectures (CNN-3x) achieve better model performance.**
        ''')
    
    with tab5: # Augmentation
        st.write("In this section, we evaluated the impact of augmentation on model performance.")
        st.image("src/visualization/CNN/4_CNN_Augmentation_table.png", use_container_width=True)
        st.image("src/visualization/CNN/4_CNN_Augmentation_graph.png", use_container_width=True)

        with st.expander("Confusion matrix"):
            st.image("src/visualization/CNN/4_CNN_Augmentation_cm.png", use_container_width=True)

        st.markdown('''
        **Summary**
        - Without augmentation (Model C), training accuracy is higher, but it seems there is overfitting, as the validation accuracy is slightly lower than the training accuracy.
        - With augmentation (Model D), training and validation accuracy are lower, and loss is higher, but the model generalizes better, as shown by improved test accuracy (0.85 vs. 0.79).
        - Data augmentation increases training time (335 ms/step vs. 55 ms/step)
        - **Data augmentation enhances model generalization, by reducing overfitting but significantly increases computational costs.**
        ''')
    
    with tab6: # CNN layer
        st.write("In this section, we evaluated the impact of the number of convoluted layers on model performance.")
        st.image("src/visualization/CNN/5_CNN_layer_table.png", use_container_width=True)
        st.image("src/visualization/CNN/5_CNN_layer_graph.png", use_container_width=True)

        with st.expander("Confusion matrix"):
            st.image("src/visualization/CNN/5_CNN_layer_cm.png", use_container_width=True)

        st.markdown('''
        **Summary**
        - Adding more layers (3x compared to 2x) enhances the model's ability to learn and generalize, as reflected in higher validation accuracy and lower validation loss.
        - The trade-off is increased computational cost (longer step time) and potential overfitting, as the 3x layers model achieves perfect training accuracy.
        - Test accuracy (C:0.79 vs E: 0.85) confirms that the 3x layers model generalizes better to unseen data
        - **Deeper architecture (3x layers) improves generalization and test accuracy (0.85 vs. 0.79) at a modest computational cost, making it a better choice for complex classification tasks.**
        ''')


elif page == pages[4]:
    st.write("### Model: Transfer Learning")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Transfer Learning", "All layers frozen", "Unfreezing of layers","Learning rate", "Image size", "Fine tuning", "Pre-trained PyTorch"])
    
    with tab1:
        st.write("After the basic CNN modelling, we incorporated pretrained models into our training that were already designed to be used in visual object recognitions. We considered some of the most common models and applied different parameter tuning to the ones that gave the best results.")
        st.write("**Pre-trained models**")
        st.markdown("· MobileNetV2")
        st.markdown("· VGG16")
        st.markdown("· ResNet 101")
        st.markdown("· ResNet 50")
        st.markdown("· EfficientNetV2")
        st.write("\n")
        # Parameters
        st.write("**Initial Parameters**")
        params = {
        'Dataset size': ['70K'],
        'Augmentation': ['No'],
        'Learning rate': ['0.001'],
        'Image size': ['256x256'],
        'Batch size': [32],
        'Number of Epochs': [50],
        'Test images': [33]
        }

        # Create DataFrame
        parameters = pd.DataFrame(params)
        st.markdown(parameters.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        st.write("\n")
        st.write("\n")
        st.write("\n")
        st.write("\n")

    

    with tab2: # Transfer Learning  # All layers frozen (Niels)
             
        st.write("In this section, we started training our models with all the layers from the pretrained models kept frozen, thus, the weights were preserved along the whole training.")         
        
        # MobileNetV2
        st.markdown("<h2 style='text-align: center; color: green;'>MobileNetV2 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/MobileNetV2_13-01-2025_Niels.png")
        st.write("\n")

        # VGG16
        st.markdown("<h2 style='text-align: center; color: green;'>VGG16 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/TL_VGG16_13-01-2025_Niels.png")
        st.write("\n")

        # ResNet 101
        st.markdown("<h2 style='text-align: center; color: green;'>ResNet 101 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/ResNet101_13-01-2025_Niels.png")
        st.write("\n")
        
        # ResNet 50
        st.markdown("<h2 style='text-align: center; color: green;'>ResNet 50 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/ResNet50_17-01-2025_Niels.png")
        st.markdown("""<span style='color: red;'>*Only 10 epochs were run on this model</span>""", unsafe_allow_html=True)
        st.write("\n")

        # EfficientNetV2-L
        st.markdown("<h2 style='text-align: center; color: green;'>EfficientNetV2-L </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Frozen_TL/EfficientNetV2L_Niels_13.01.2025.png")
        st.markdown("""<span style='color: red;'>*Only 22 epochs were run on this model</span>""", unsafe_allow_html=True)
        st.write("\n")

        # Summary of metrics
        st.markdown("**Metrics summary**")
        data = {
        'Model': [
        'MobileNetV2', 'VGG16', 
        'ResNet 101', 'ResNet 50', 'EfficientNetV2-L'
        ],
        'Accuracy': [0.99, 0.97, 0.51, 0.39, 0.26],
        'Loss': [0.08, 0.09, 1.60, 2.00, 2.58],
        'Val-accuracy': [0.97, 0.96, 0.56, 0.48, 0.39],
        'Val-loss': [0.23, 0.14, 1.51, 1.75, 2.08],
        'Test Prediction Accuracy': [0.97, 0.97, "n.d.", "n.d.", "n.d."]
        }

        # Create DataFrame with the updated values
        df = pd.DataFrame(data)
        df.set_index ('Model', inplace=True)       
        st.dataframe(df)
        st.write("\n")
        st.write("\n")

    with tab3: # LAYER UNFREEZING (Lara)
        
        st.write("After considering the performance of the various pre-trained models, we picked the best two models (MobileNetV2 and VGG16) to analyse further how unfreezing layers affects the performance of the classification. In this section, we first unfroze all layers of the pre-trained model and after that, we repeated the training, but only unfreezing the last block of layers and compared performance.")         
        # MobileNetV2
        st.markdown("<h2 style='text-align: center; color: green;'>MobileNetV2 </h2>", unsafe_allow_html=True)
        st.markdown("**1) All layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_unfrozen.png")
        st.write("\n")

        
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_unfrozen_CM.png")
        st.write("\n")
        st.write("\n")

        st.markdown("**2) Only last block of layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_1Block_unfrozen.png")
        st.write("\n")

                
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_MobileNet_1Block_unfrozen_CM.png")
        st.write("\n")
        st.write("\n")

        # VGG16
        st.markdown("<h2 style='text-align: center; color: green;'>VGG16 </h2>", unsafe_allow_html=True)
        st.markdown("**1) All layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen.png")
        st.write("\n")

        
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_CM.png")
        st.write("\n")
        st.write("\n")

        st.markdown("**2) Only last block of layers unfrozen**")
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_1Block_unfrozen.png")
        st.write("\n")

                
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_1Block_unfrozen_CM.png")
        st.write("\n")
        st.write("\n")

        # Summary of metrics
        st.markdown("**Metrics summary**")
        data = {
        'Model': [
        'MobileNetV2 unfrozen', 'MobileNetV2 partly frozen', 
        'VGG16 unfrozen', 'VGG16 partly frozen'
        ],
        'Accuracy': [0.99, 0.99, 0.03, 0.99],
        'Loss': [0.02, 0.02, 3.64, 0.04],
        'Val-accuracy': [0.99, 0.99, 0.03, 0.96],
        'Val-loss': [0.05, 0.05, 3.64, 0.15],
        'Test Prediction Accuracy': [1.00, 0.93, 0.09, 0.88]
        }

        # Create DataFrame with the updated values
        df = pd.DataFrame(data)
        df.set_index ('Model', inplace=True)       
        st.dataframe(df)
        st.write("While there was only a small difference in performance between the MobileNetV2 models, regardless of whether all layers were unfrozen or not, the VGG16 model showed a dramatic improvement when only the last block of layers was unfrozen.")
        st.write("\n")

    with tab4:  # Modification of learning rate
              
        st.write("Considering the bad performance given by the model with VGG16 with all the layers unfrozen, we decided to reduce the learning rate from 0.001 to 0.0001.")
        st.markdown("<h2 style='text-align: center; color: green;'>VGG16 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_lr10e-4.png")
        st.write("\n")

        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_lr10e-4_CM.png")
        st.write("\n")

    with tab5: # Modification of the image size
        st.write("In this section we changed the input image size from 256x256 to 224x224 to see how this could change the training for the model with VGG16, while keeping the learning rate of 0.001, which gave the bad results.")
        st.markdown("<h2 style='text-align: center; color: green;'>VGG16 </h2>", unsafe_allow_html=True)
        st.markdown("""<div style='text-align: center;'>Metrics history</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_224.png")
        st.write("\n")        
        
        st.markdown("""<div style='text-align: center;'>Confusion matrix</div>""", unsafe_allow_html=True)
        st.image("src/visualization/Transfer_Learning_param_tests/TL_VGG16_unfrozen_224_CM.png")
        st.write("\n")

    
    with tab6: # Fine tuning with VGG16 
        st.write("In this section we went deeper into the optimization of the model with VGG16. From the previous trainings we observed dramatic improvements with some parameter changes, as we see in the following table:")
        # Summary of previous metrics
        st.markdown("**Previous metrics summary**")
        data2 = {
        'Model': [
            'VGG16 unfrozen', 'VGG16 partly frozen', 'VGG16 unfrozen lr 1E-4', 'VGG16 unfrozen size 224x224'
        ],
        'Accuracy': [0.03, 0.99, 0.99, 1.00],
        'Loss': [3.64, 0.04, 0.05, 0.00],
        'Val-accuracy': [0.03, 0.96, 0.96, 1.00],
        'Val-loss': [3.64, 0.15, 0.14, 0.01],
        'Test Prediction Accuracy': [0.09, 0.88, 0.97, 0.97]
        }
        # Create DataFrame with the updated values
        df2 = pd.DataFrame(data2)
        df2.set_index ('Model', inplace=True)       
        st.dataframe(df2)
        st.write("\n")
        st.write("\n")
        
        st.markdown('''
        After learning the parameters that worked well, we decided to fine-tune the modelling part until we optimized the training with VGG16.  
        The changed parameters that gave the best performance were:

        **1. Image Size Adjustment:**  
        The input image size was set to 224 by 224 pixels, which matches the dimensions of the pretrained ImageNet dataset. This adjustment enabled better alignment between our dataset and the pretrained weights, significantly improving the model's performance.

        **2. Learning Rate Optimization**  
        A lower learning rate was used, which proved to be an essential factor in stabilizing the training process and improving accuracy.

        **3. Progressive Fine-Tuning Approach:**  
        We adopted a progressive fine-tuning strategy to train the model effectively. This approach involved the following steps:  
        - **Step 1:** The model was trained for 5 epochs with all layers frozen, allowing the newly added layers to adapt to the dataset without disrupting the pretrained feature extractor.  
        - **Step 2:** All layers (or the last block) were unfrozen, and the model was trained for an additional 45 epochs using a smaller learning rate. This step fine-tuned the higher-level features to align better with our dataset's specific requirements.
        ''')

        
        ###########################################
        st.image("src/visualization/CNN/6_VGG16_table.png", use_container_width=True)
        st.image("src/visualization/CNN/6_VGG16_graph.png", use_container_width=True)

        with st.expander("Confusion matrix"):
            st.image("src/visualization/CNN/6_VGG16_cm.png", use_container_width=True)

        container = st.container(border=True)
        container.write("**The VGG16 architecture, with all layers unfrozen, demonstrated the best performance, achieving a high level of prediction accuracy.**")
        st.write("\n")

        st.markdown("**Evaluation with test datasets of different sizes**")

        with st.expander("Test dataset"):
            st.image("src/visualization/CNN/TestDataset_all.png", use_container_width=True)
        
        with st.expander("Test accuracy"):
            st.image("src/visualization/CNN/Test-accuracy_F-3.png", use_container_width=True)
        
        st.write("\n")   

        # Prediction with two test dataset
        df = pd.read_csv("src/visualization/CNN/F_3-3_VGG16_twoStep-all-unfrozen_predictions_1.csv", header=1)
        st.write("**Prediction with test dataset (33 images):**")
        st.dataframe(df)

        df2 = pd.read_csv("src/visualization/CNN/F_3-3_VGG16_twoStep-all-unfrozen_predictions_2.csv", header=1)
        st.write("**Prediction with larger test dataset (283 images):**")
        st.dataframe(df2)

    with tab7: # Pre-trained models with Pytorch (Yannik)
        plantpad_url = "http://plantpad.samlab.cn"
        st.write("""We used a pre-trained model from [www.plantpad.samlab.cn](%s) which provides models for plant disease diagnosis.
                 These models have been trained on image data (421,314 images) consisting of 63 plant species and 310 kinds of plant diseases. 
                 We employed a ResNet50 model for transfer learning by using it as the base model for feature extraction on which we added a 3-layer classifier CNN.""" % plantpad_url)
        st.write("")
        st.write("")
        st.write("We again used the same inital training parameters as for the TL models in Keras and kept the base model layers frozen:")
        st.markdown(parameters.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")
        st.image("src/visualization/Transfer_Learning_PyTorch_ResNet50/ResNet50_all-frozen_lr-1e-3.png")
        st.write("")
        st.write("")
        st.write("""Although the training and validation accuracies are above 0.98, the validation loss shows high fluctuations instead
                 of decreasing. This suggests that the model is not learning properly. As a next step, we decreased the learning rate to 1e-4.""")
        st.write("")
        st.write("")
        st.image("src/visualization/Transfer_Learning_PyTorch_ResNet50/ResNet50_all-frozen_lr-1e-4.png")
        st.write("")
        st.write("")
        st.write("""Lowering the learning rate improves the validation loss on an absolute scale but the fluctuations during training 
                 are still observable. At this point, we would need to look again at the model's architecture and possibly compare these 
                 results with other transfer learning models. However, the performance of the best model is very good. On the (admittedly small) 
                 test set of 33 images, it predicted all classes correctly. The confusion matrix for the validation set shows only one-digit entries 
                 on the off-diagonal. It predicted the class “Corn_(maize)__healthy” worst where it confused it nine times with “Strawberry__healthy”. 
                 From the total 17,572 images in the validation set, the model failed to predict the correct class in 43 cases, corresponding to an 
                 accuracy of 0.997.
        """)
        st.write("")
        st.write("")
        st.image("src/visualization/Transfer_Learning_PyTorch_ResNet50/ResNet50_all-frozen_lr-1e-4_confusion_matrix.png")

####################
# MODEL INTERPRET  #
####################


elif page == pages[5]:
    st.write("## Model Interpretability")
    st.write("### Why It Matters")
    st.write("""
        - **Trust and Transparancy**: Understanding why a model makes certain predictions increases user trust.
        - **Error Analysis**: Interpretability helps identify model biases, misclassifications, and areas for improvement. 
        - **Debugging Models**: Insights into model decisions enable developers to identify issues in data, architecture, or training.
                 """)
    st.write("")
    st.write("")
    st.write("### Gradient-weighted Class Activation Mappping (Grad-CAM)")
    st.write("**What is Grad-CAM?**")
    st.write("""Grad-CAM is a technique that visualizes the regions of an input image that contribute most to the model's prediction. 
             It provides class-specific heatmaps overlaid on the original image. Understanding why a model makes certain predictions 
             increases user trust.
             """)
    st.write("**How it works:**")
    st.write("""
        - Choosing the intermediate output (feature map) of a convolutional layer in the model.
        - Computing the gradient of the model’s final output with respect to the feature map. 
        - all regions are then weighted and combined to give a heatmap which is projected onto the input image.
            """)
    st.write("""**High gradients** in a certain image region mean that a small change in the feature map will **significantly affect** the model **output**.
             The heatmap thus highlights spatially relevant features in an image and enhances trust by showing **where the model "looks"** to make its decision.
             """)
    st.write("")
    st.image("src/visualization/Grad-CAM.png")


####################
# CONCLUSION       #
####################


elif page == pages[6]:
    st.write("### Conclusion")
   
    tab1, tab2, tab3 = st.tabs(["Results", "GradCam" ,"Future perspectives"])
    
    with tab1:
        st.write('''
    This study explored the evolution of image classification models, from basic CNNs to advanced transfer learning techniques. 
    The use of pre-trained models like MobileNetV2, ResNet50 and VGG16, combined with fine-tuning and layer unfreezing, led to significant performance improvements, achieving near-perfect accuracy. 
    The results show that model performance depends on factors like learning rates, layer freezing, and optimization, with transfer learning offering the best results.\n
    The 3 best models obtained, according to the metrics and preformance, were:    
    ''')
        
        # Summary of metrics
        st.markdown("**Metrics summary**")
        data_conclusion = {
        'Model': [
        'MobileNetV2', 'VGG16', 
        'ResNet 50'
        ],
        'Image size': ["256 x 256", "224 x 224", "256 x 256"],
        'Learning rate': ["1E-3", "1E-4", "1E-4"],
        'Freezing': ["All unfrozen", "2-step frozen", "All frozen"],
        'Train-acc': [0.99, 0.9993, 1.00],
        'Train-loss': [0.02, 0.003, 0.01],
        'Val-acc': [0.99, 0.997, 1.00],
        'Val-loss': [0.05, 0.008, 0.01],
        'Test-acc': [1.00, 1.00, 1.00]
        }

        # Create DataFrame with the updated values
        df = pd.DataFrame(data_conclusion)
        df.set_index ('Model', inplace=True)       
        st.dataframe(df)
        st.write("\n")
        st.write("\n")
   
        st.write(" Initially, basic CNN models, such as the sequential 2-layer networks, displayed relatively limited performance. Despite slight improvements across varying learning rates and input image sizes (e.g., 224x224 vs. 256x256), these models struggled with high validation losses, often indicating overfitting or insufficient generalization. Models like the Sequential CNN 2-layer with a learning rate of 0.0001 and a 224x224 input size achieved moderate validation accuracy but still left much to be desired.")
        st.write(" The next step in the modeling evolution involved transfer learning with pre-trained architectures, such as VGG16, MobileNetV2, and ResNet101, among others, which significantly improved model performance. These models started with the ImageNet weights and frozen layers to retain the learned features from the large-scale dataset, with the best results coming from MobileNetV2 and VGG16, especially when the learning rate was finely tuned. The models with the lowest validation loss and highest validation accuracy were those that used MobileNetV2 (frozen), achieving a near-perfect 0.97 accuracy and low validation loss (0.23), indicating their superior generalization capability over CNN-based models.")
        st.write(" The next refinement occurred through unfreezing layers in the transfer learning models, allowing fine-tuning of the network on the specific dataset. Models like VGG16 unfrozen and MobileNetV2 unfrozen showed impressive performance, particularly when combined with lower learning rates (e.g., 0.0001), which helped the model converge to an optimal solution. The best performing models in terms of both prediction and validation accuracy were those with fully or partly unfrozen layers, such as VGG16 unfrozen V1-2, where a two-step layer unfreezing was applied. This model achieved a perfect validation accuracy of 1.00 with minimal loss.")
        st.write(" Finally, we applied transfer learning using a model that was already pre trained on plant images, ResNet50, to see how accurate this model could be with our given dataset. For that, we changed the framework to Pytorch instead of Tensorflow/Keras and compared the performance. Again, reducing the learning rate, gave a dramatic improvement of the metrics performance. However, in this case, all the blocks of layers were kept frozen.")

    with tab2:
        st.write('''
    We chose GradCAM as the interpretation method because it provides clear, visual insights. Its heatmaps highlight the importance of each pixel in relation to the predicted class in a given layer by adjusting the intensity of the pixels. 
    This makes it easier for users to understand and trust the model's decision-making process, increasing confidence in its interpretability.\n
    The following image shows two examples of the gradients obtained by the camera in a model using VGG16:

    ''')
        st.image("src/visualization/Conclusion/GC_VGG16_example.png")

        st.write('''
    The observation of the GradCam images suggests that models can have significantly different focus areas. While some models, like VGG16, focus on the object for feature extraction, others may focus more on the background of the image. 
    These differing approaches can lead to varying results depending on the specific image being classified.
    ''')
        st.write("**Therefore, we can conclude that here is no one-size-fits-all approach that guarantees the best results. The effectiveness of each method depends on how well it is optimized and the resources required for it.**")
    
    with tab3: 
        st.write("This project provided valuable experience in utilizing deep learning and convolutional neural networks (CNNs) for image classification tasks. We achieved excellent results with a small test dataset. However, the model still requires further refinement to effectively handle and predict large datasets. There are several potential improvements that could enhance model performance. One such option would be the introduction of data transformations, such as thresholding or Canny filtering. Although we mentioned this approach in the data exploration report, we were unable to test it due to programming resource constraints.")

####################
# UPLOAD IMAGE     #
####################


elif page == pages[7]:
    plantpad_url = "http://plantpad.samlab.cn"
    st.write("### Predict your plant")
    st.write("")
    st.write("You can select between three final models:")
    st.markdown("- ResNet50 (PyTorch) based on the pre-trained model from [www.plantpad.samlab.cn](%s)" % plantpad_url)
    st.markdown("- VGG16 (Keras): all layers unfrozen, fine-tuned with a learning rate of 1e-5 and img size of 224x224")
    st.markdown("- MobileNetV2 (Keras): all layers unfrozen, a learning rate of 1e-3 and img size of 256x256")


    # Dropdown menu for selecting a trained model
    model_files = [f for f in os.listdir("src/models/") if f.endswith(".keras") or f.endswith(".pth")]
    selected_model_file = st.selectbox("Select a trained model:", model_files)

    # Drag-and-drop file uploader for image
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",  width=300)

        # Load the selected model
        model_path = os.path.join("src/models", selected_model_file)

        # Calculate predictions and handle confidence probabilities
        if selected_model_file.endswith(".keras"):
            model = load_keras_model(model_path)
            preprocessed_image = preprocess_image(image, model)
            predictions = model.predict(preprocessed_image)  # No softmax needed
            predicted_idx = np.argmax(predictions[0])
            top_predictions = predictions[0].argsort()[-5:][::-1]
            probabilities = predictions[0]  # Use raw probabilities as is
        elif selected_model_file.endswith(".pth"):
            model = load_pytorch_model(model_path)
            preprocessed_image = preprocess_image(image, model)
            predictions = model(preprocessed_image).detach().numpy()
            probabilities = np.exp(predictions[0]) / np.sum(np.exp(predictions[0]))  # Apply softmax for PyTorch
            predicted_idx = np.argmax(probabilities)
            top_predictions = probabilities.argsort()[-5:][::-1]


        # Load plant and disease class indices
        plant_indices = load_class_indices("src/streamlit/class_plant_indices.json")
        disease_indices = load_class_indices("src/streamlit/class_disease_indices.json")

        # Display predicted class name in a table
        st.subheader("Model Predictions")
        predicted_plant = plant_indices[str(predicted_idx)]  # Assuming diseases are indexed per plant
        predicted_disease = disease_indices[str(predicted_idx)]
        st.write("**Predicted Plant Type**:", predicted_plant)
        st.write("**Predicted Disease Type**:", predicted_disease)

        # Checkbox for displaying top predictions
        display_top_predictions = st.checkbox("Display top predictions")

        # Display top 5 predictions
        if display_top_predictions:
            st.subheader("Top Predictions")
            sorted_indices = top_predictions  # Already sorted
            seen_combinations = set()  # To avoid duplicate combinations
            top_pred_table = {
                "Plant Type": [],
                "Disease Type": [],
                "Confidence": []
            }
            for idx in sorted_indices:
                plant = plant_indices[str(idx)]
                disease = disease_indices[str(idx)]
                combination = (plant, disease)
                if combination not in seen_combinations:  # Avoid duplicates
                    seen_combinations.add(combination)
                    confidence = probabilities[idx]
                    top_pred_table["Plant Type"].append(plant)
                    top_pred_table["Disease Type"].append(disease)
                    top_pred_table["Confidence"].append(f"{confidence:.2%}")
                if len(top_pred_table["Plant Type"]) == 5:  # Stop at top 5
                    break

            st.table(top_pred_table)


    # Checkbox for Grad-CAM
    display_grad_cam = st.checkbox("Display Grad-CAM")

    # Generate and display Grad-CAM if selected
    if display_grad_cam:
        st.subheader("Grad-CAM Visualization")

        gradcam_model_path = f"src/models/plain_architectures/plain_{selected_model_file}"
        if selected_model_file.endswith(".keras"):
            gradcam_model = load_keras_model(gradcam_model_path)
            grad_cam_image = generate_grad_cam_keras(gradcam_model, image, layer_name=None)
            st.image(grad_cam_image, caption=f"Grad-CAM of {selected_model_file}", width=300)


        elif selected_model_file.endswith(".pth"):
            gradcam_model = load_pytorch_model(model_path)
            grad_cam_image = generate_grad_cam_pytorch(gradcam_model, image)
            st.image(grad_cam_image, caption=f"Grad-CAM of {selected_model_file}", width=300)
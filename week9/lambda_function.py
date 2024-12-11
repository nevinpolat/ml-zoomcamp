import json
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='curly-vs-straight.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_content, target_size=(200, 200)):
    """
    Preprocess the image: resize, convert to RGB, rescale pixel values.
    """
    # Open the image from bytes
    img = Image.open(BytesIO(image_content)).convert('RGB')
    # Resize the image to the target size
    img = img.resize(target_size, Image.NEAREST)
    # Convert to numpy array and rescale pixel values to [0, 1]
    img_array = np.array(img) / 255.0
    # Expand dimensions to match model input shape (1, 200, 200, 3)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def lambda_handler(event, context):
    """
    Lambda function handler to process the image and return prediction.
    """
    # Extract image URL from the event
    image_url = event.get('image_path', '')
    if not image_url:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No image_path provided'})
        }
    
    # Download the image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_content = response.content
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Failed to download image: {str(e)}'})
        }
    
    # Preprocess the image with the desired target size
    try:
        img_array = preprocess_image(image_content, target_size=(200, 200))
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Failed to preprocess image: {str(e)}'})
        }
    
    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run the inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = float(output_data[0][0])
    
    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': prediction})
    }

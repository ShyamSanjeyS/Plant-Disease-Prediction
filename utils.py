import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes=None):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location=device)
        
        if num_classes is None:
            if 'fc.weight' in state_dict:
                num_classes = state_dict['fc.weight'].shape[0]
            else:
                raise RuntimeError("Cannot determine number of classes from saved model")
        
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
            model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e2:
                raise RuntimeError(f"Failed to load model state dict: {str(e2)}")
        
        model = model.to(device)
        model.eval()
        
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits = None
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {str(e)}")

def preprocess_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_size = 299
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0).to(device)
        
        return tensor
        
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image: {str(e)}")

def predict(image_tensor, model, class_names):
    try:
        model.eval()
        
        with torch.no_grad():
            output = model(image_tensor)
            
            if isinstance(output, tuple):
                output = output[0]
            
            probs = torch.nn.functional.softmax(output, dim=1)
            probs_np = probs.cpu().numpy()[0]
            
            pred_idx = probs_np.argmax()
            
            if pred_idx >= len(class_names):
                pred_idx = len(class_names) - 1
            
            predicted_class = class_names[pred_idx]
            
            return predicted_class, probs_np
            
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

def get_model_info():
    info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available(),
        'torch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    return info

def validate_image(image_path_or_pil):
    try:
        if isinstance(image_path_or_pil, str):
            if os.path.exists(image_path_or_pil):
                image = Image.open(image_path_or_pil)
            else:
                return None
        else:
            image = image_path_or_pil
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if image.size[0] < 32 or image.size[1] < 32:
            print("Warning: Image is very small, results may be poor.")
        
        return image
        
    except Exception as e:
        print(f"Error validating image: {str(e)}")
        return None

def get_layer_names(model):
    layer_names = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            layer_names.append(name)
    return layer_names

def check_model_compatibility(model_path, expected_classes):
    try:
        if not os.path.exists(model_path):
            return False
        
        state_dict = torch.load(model_path, map_location='cpu')
        
        if 'fc.weight' in state_dict:
            fc_weight_shape = state_dict['fc.weight'].shape
            if fc_weight_shape[0] != expected_classes:
                print(f"Warning: Model expects {fc_weight_shape[0]} classes, but {expected_classes} provided")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error checking model compatibility: {str(e)}")
        return False

def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
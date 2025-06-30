import torch
import cv2
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

def generate_gradcam(model, input_tensor, target_layer, class_idx):
    gradients = []
    activations = []

    def save_gradient_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0])

    def save_activation_hook(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0])
        else:
            activations.append(output)

    try:
        handle_grad = target_layer.register_full_backward_hook(save_gradient_hook)
        handle_act = target_layer.register_forward_hook(save_activation_hook)
    except AttributeError:
        handle_grad = target_layer.register_backward_hook(save_gradient_hook)
        handle_act = target_layer.register_forward_hook(save_activation_hook)

    try:
        input_tensor.requires_grad_(True)
        
        model.zero_grad()
        output = model(input_tensor)
        
        if output.dim() < 2 or output.size(0) == 0:
            raise ValueError("Invalid model output")
            
        if class_idx >= output.size(1):
            class_idx = output.size(1) - 1
            
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)

        if not gradients or not activations:
            raise RuntimeError("Failed to capture gradients or activations")

        grads = gradients[0].cpu().detach().numpy()
        acts = activations[0].cpu().detach().numpy()
        
        if grads.ndim == 4:
            grads = grads[0]
        if acts.ndim == 4:
            acts = acts[0]

        weights = np.mean(grads, axis=(1, 2))
        
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * acts, axis=0)
        
        cam = np.maximum(cam, 0)
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        cam = cv2.resize(cam, (299, 299))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        return heatmap

    except Exception as e:
        print(f"Error in generate_gradcam: {str(e)}")
        default_cam = np.zeros((299, 299))
        return cv2.applyColorMap(np.uint8(255 * default_cam), cv2.COLORMAP_JET)
        
    finally:
        try:
            handle_grad.remove()
            handle_act.remove()
        except:
            pass

def overlay_gradcam(image, heatmap, alpha=0.4):
    try:
        image_np = np.array(image.resize((299, 299)))
        
        if heatmap.shape[:2] != image_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            if len(heatmap.shape) == 3 and heatmap.shape[2] == 3:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlayed = cv2.addWeighted(image_np, 1-alpha, heatmap, alpha, 0)
        
        return overlayed
        
    except Exception as e:
        print(f"Error in overlay_gradcam: {str(e)}")
        return np.array(image.resize((299, 299)))

def generate_gradcam_multiple_layers(model, input_tensor, target_layers, class_idx):
    heatmaps = {}
    
    for layer_name, layer in target_layers.items():
        try:
            heatmap = generate_gradcam(model, input_tensor, layer, class_idx)
            heatmaps[layer_name] = heatmap
        except Exception as e:
            print(f"Failed to generate Grad-CAM for layer {layer_name}: {str(e)}")
            default_cam = np.zeros((299, 299))
            heatmaps[layer_name] = cv2.applyColorMap(np.uint8(255 * default_cam), cv2.COLORMAP_JET)
    
    return heatmaps
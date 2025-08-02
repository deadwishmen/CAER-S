import argparse
import torch
import os
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
import cv2
import random



def main(config):
    logger = config.get_logger('detect')

    config['test_loader']['args']['batch_size'] = 1

    data_loader = config.init_obj('test_loader', module_data)

    model = config.init_obj('arch', module_arch)

    logger.info(model)

    logger.info('loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'), weights_only=False)

    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # --- EigenCAM ---

    grad_cam_config = config['grad_cam']
    output_dir = grad_cam_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    visualize_on = grad_cam_config.get('visualiza_on', 'body')

    # Get target layers from config
    try:
        target_layers_str = grad_cam_config['target_layers']
        # using eval path layers children
        target_layers = [eval(f"model.{target_layers_str}")]
        logger.info(f"Using target layers: {target_layers_str}")
    except Exception as e:
        logger.error(f"Error parsing target layers: {e}")
        return

    # Wrapper classes cho từng input type
    class FaceModelWrapper(torch.nn.Module):
        def __init__(self, model, context_sample, body_sample):
            super().__init__()
            self.model = model
            self.context_sample = context_sample
            self.body_sample = body_sample
            
        def forward(self, face):
            return self.model(face, self.body_sample, self.context_sample)

    class BodyModelWrapper(torch.nn.Module):
        def __init__(self, model, face_sample, context_sample):
            super().__init__()
            self.model = model
            self.face_sample = face_sample
            self.context_sample = context_sample
            
        def forward(self, body):
            return self.model(self.face_sample, body, self.context_sample)

    class ContextModelWrapper(torch.nn.Module):
        def __init__(self, model, face_sample, body_sample):
            super().__init__()
            self.model = model
            self.face_sample = face_sample
            self.body_sample = body_sample
            
        def forward(self, context):
            return self.model(self.face_sample, self.body_sample, context)

    class_names = config['class_names']
    
    # Số lượng mẫu random muốn xử lý
    num_random_samples = grad_cam_config.get('num_samples', 10)  # Mặc định 10 mẫu
    
    # Tạo list các index random
    total_samples = len(data_loader)
    if num_random_samples >= total_samples:
        selected_indices = list(range(total_samples))
    else:
        selected_indices = random.sample(range(total_samples), num_random_samples)
    
    logger.info(f"Selected {len(selected_indices)} random samples from {total_samples} total samples")
    logger.info(f"Selected indices: {selected_indices}")

    logger.info(f"Starting detection. Visualizing CAM on '{visualize_on}' image ...")
    
    # --- Vòng lặp xử lý dữ liệu ---
    for i, (data, target) in enumerate(tqdm(data_loader)):
        # Chỉ xử lý các sample được chọn
        if i not in selected_indices:
            continue
            
        face, body, context, target = data['face'].to(device), data['body'].to(device), data['context'].to(device), target.to(device)

        input_tensor_tuple = (face, body, context)

        output = model(*input_tensor_tuple)

        pred_index = torch.argmax(output, dim=1)[0].item()
        true_index = target[0].item()

        # Tạo wrapper model tương ứng với visualize_on
        if visualize_on == 'face':
            wrapped_model = FaceModelWrapper(model, context, body)
            input_for_cam = face
            input_image_tensor = face[0]
        elif visualize_on == 'context':
            wrapped_model = ContextModelWrapper(model, face, body)
            input_for_cam = context
            input_image_tensor = context[0]
        else:  # body
            wrapped_model = BodyModelWrapper(model, face, context)
            input_for_cam = body
            input_image_tensor = body[0]

        # Khởi tạo EigenCAM cho wrapper model cụ thể
        cam = EigenCAM(model=wrapped_model, target_layers=target_layers)
        
        # Tạo heatmap
        grayscale_cam = cam(input_tensor=input_for_cam, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # --- Phần trực quan hóa và lưu ảnh ---
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if input_image_tensor.shape[0] == 1:
            input_image_tensor = input_image_tensor.repeat(3, 1, 1)

        rgb_img = input_image_tensor.cpu().numpy().transpose((1, 2, 0))
        rgb_img = std * rgb_img + mean
        rgb_img = np.clip(rgb_img, 0, 1)
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        original_img_bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        pred_class_name = class_names[pred_index]
        true_class_name = class_names[true_index]
        filename_prefix = f"sample_{i}_true_{true_class_name}_pred_{pred_class_name}"
        
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_original_{visualize_on}.jpg"), original_img_bgr)
        cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_cam_on_{visualize_on}.jpg"), visualization_bgr)

    logger.info(f"EigenCAM generation finished. Results saved in '{output_dir}'.")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    
    config = ConfigParser.from_args(args)
    main(config)
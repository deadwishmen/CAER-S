import argparse
import torch
import os
import numpty as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from pytorch_grad_cam import GradCAM
from prtorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
import cv2



def main(config):
    logger = config.get_logger('detect')

    config['train_loader']['args']['batch_size'] = 1

    data_loader = config.init_obj('test_loader', module_data)

    model = config.init_obj('arch', module_arch)

    logger.info(model)

    logger.info('loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume, wieghts_only=False)

    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # --- Grad-CAM ---

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

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=torch.cuda.is_available())
    class_names = data_loader.dataset.classes

    logger.info(f"Starting detection. Visualinzing CAM on '{visualize_on}' image ...")
    
    # --- Loop through the data loader ---
    for i, (data, target) in enumerate(tqdm(data_loader)):
            face, body, context, target = data['face'].to(device), data['body'].to(device), data['context'].to(device), target.to(device)

            # Với mô hình đa đầu vào, input_tensor phải là một tuple
            input_tensor_tuple = (face, body, context)

            # Lấy output từ model để xác định class dự đoán
            output = model(*input_tensor_tuple)
            pred_index = torch.argmax(output, dim=1).item()
            true_index = target.item()

            # Tạo heatmap. target_category=None sẽ tự động dùng class có score cao nhất
            grayscale_cam = cam(input_tensor=input_tensor_tuple, target_category=None)
            grayscale_cam = grayscale_cam[0, :]

            # --- 5. Trực quan hóa và lưu ảnh ---
            if visualize_on == 'face':
                input_image_tensor = face[0]
            elif visualize_on == 'context':
                input_image_tensor = context[0]
            else: # Mặc định là body
                input_image_tensor = body[0]
                
            # De-normalize ảnh để hiển thị đúng màu sắc
            # QUAN TRỌNG: Hãy thay đổi mean và std cho đúng với quá trình tiền xử lý của bạn
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            # Nếu ảnh là grayscale (ví dụ: face), mở rộng thành 3 kênh
            if input_image_tensor.shape[0] == 1:
                input_image_tensor = input_image_tensor.repeat(3, 1, 1)

            rgb_img = input_image_tensor.cpu().numpy().transpose((1, 2, 0))
            rgb_img = std * rgb_img + mean
            rgb_img = np.clip(rgb_img, 0, 1)
            
            # Chồng heatmap lên ảnh gốc
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Chuyển đổi lại sang BGR để OpenCV lưu
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            original_img_bgr = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Tạo tên file
            pred_class_name = class_names[pred_index]
            true_class_name = class_names[true_index]
            filename_prefix = f"sample_{i}_true_{true_class_name}_pred_{pred_class_name}"
            
            # Lưu các ảnh
            cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_original_{visualize_on}.jpg"), original_img_bgr)
            cv2.imwrite(os.path.join(output_dir, f"{filename_prefix}_cam_on_{visualize_on}.jpg"), visualization_bgr)

    logger.info(f"Grad-CAM generation finished. Results saved in '{output_dir}'.")

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

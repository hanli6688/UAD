import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import torch
import argparse
import json
import cv2
import numpy as np

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from PIL import Image

# Local modules
from tools import write2csv, setup_seed, Logger
from dataset import get_data, dataset_dict
from method.uad_trainer import UAD_Trainer

setup_seed(111)


def test(args):
    assert os.path.isfile(args.ckt_path), \
        f"Invalid checkpoint path: {args.ckt_path}"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = Logger('log_test.txt')

    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    # =========================
    # Load config
    # =========================
    config_path = os.path.join('./model_configs', f'{args.model}.json')
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage*2, substage*3, substage*4]

    # =========================
    # Initialize UAD
    # =========================
    model = UAD_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=args.image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters,
        use_idag=False   # 测试阶段关闭
    ).to(device)

    model.load(args.ckt_path)
    model.eval()

    # =========================================================
    # =============== Dataset Testing ==========================
    # =========================================================
    if args.testing_model == 'dataset':

        assert args.testing_data in dataset_dict.keys()

        save_root = args.save_path
        csv_root = os.path.join(save_root, 'csvs')
        image_root = os.path.join(save_root, 'images')

        os.makedirs(csv_root, exist_ok=True)
        os.makedirs(image_root, exist_ok=True)

        csv_path = os.path.join(csv_root, f'{args.testing_data}.csv')
        image_dir = os.path.join(image_root, args.testing_data)

        test_data_cls_names, test_data, _ = get_data(
            dataset_type_list=args.testing_data,
            transform=model.preprocess,
            target_transform=model.transform,
            training=False
        )

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False
        )

        metric_dict = evaluate_uad(
            model,
            test_loader,
            test_data_cls_names,
            image_dir,
            save_fig=args.save_fig
        )

        for tag, data in metric_dict.items():
            logger.info(
                '{:>15} | I-AUROC:{:.2f} | P-AUROC:{:.2f} | P-F1:{:.2f}'.format(
                    tag,
                    data['auroc_im'],
                    data['auroc_px'],
                    data['f1_px']
                )
            )

        for k in metric_dict.keys():
            write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

    # =========================================================
    # =============== Single Image Testing =====================
    # =========================================================
    elif args.testing_model == 'image':

        assert os.path.isfile(args.image_path)

        ori_image = cv2.resize(
            cv2.imread(args.image_path),
            (args.image_size, args.image_size)
        )

        pil_img = Image.open(args.image_path).convert('RGB')
        img = model.preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            anomaly_map, score_dict = model.inference(img, [args.class_name])

        anomaly_map = anomaly_map[0].cpu().numpy()

        # ===== smoothing =====
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)

        # ===== normalization =====
        anomaly_map = (anomaly_map - anomaly_map.min()) / \
                      (anomaly_map.max() - anomaly_map.min() + 1e-6)

        anomaly_map = (anomaly_map * 255).astype(np.uint8)

        heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        vis_map = cv2.addWeighted(ori_image, 0.5, heat_map, 0.5, 0)

        vis = cv2.hconcat([ori_image, vis_map])

        save_path = os.path.join(args.save_path, args.save_name)
        os.makedirs(args.save_path, exist_ok=True)

        print(f"[UAD] anomaly score = {score_dict['final']:.4f}")
        cv2.imwrite(save_path, vis)


# =========================================================
# UAD Evaluation (核心区别)
# =========================================================
@torch.no_grad()
def evaluate_uad(model, dataloader, cls_names, image_dir, save_fig=False):

    results = {}
    all_img_scores = []
    all_px_scores = []

    for imgs, _ in tqdm(dataloader):
        imgs = imgs.to(model.device)

        anomaly_map, score_dict = model.inference(imgs, cls_names)

        img_score = score_dict['final']
        px_score = anomaly_map.mean(dim=[1, 2])

        all_img_scores.append(img_score.cpu())
        all_px_scores.append(px_score.cpu())

    img_scores = torch.cat(all_img_scores)
    px_scores = torch.cat(all_px_scores)

    results['Average'] = {
        'auroc_im': img_scores.mean().item(),
        'auroc_px': px_scores.mean().item(),
        'f1_px': px_scores.mean().item()
    }

    return results


def str2bool(v):
    return v.lower() in ("yes", "true", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("UAD Test")

    parser.add_argument("--ckt_path", type=str, required=True)

    parser.add_argument("--testing_model", type=str,
                        default="dataset",
                        choices=["dataset", "image"])

    parser.add_argument("--testing_data", type=str, default="visa")

    parser.add_argument("--image_path", type=str, default="test.png")
    parser.add_argument("--class_name", type=str, default="candle")
    parser.add_argument("--save_name", type=str, default="result.png")

    parser.add_argument("--save_path", type=str, default="./workspaces")

    parser.add_argument("--model", type=str, default="ViT-L-14-336")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=518)

    parser.add_argument("--prompting_depth", type=int, default=4)
    parser.add_argument("--prompting_length", type=int, default=5)
    parser.add_argument("--prompting_type", type=str, default='SD')
    parser.add_argument("--prompting_branch", type=str, default='VL')

    parser.add_argument("--use_hsf", type=str2bool, default=True)
    parser.add_argument("--k_clusters", type=int, default=20)

    parser.add_argument("--save_fig", type=str2bool, default=False)

    args = parser.parse_args()

    test(args)
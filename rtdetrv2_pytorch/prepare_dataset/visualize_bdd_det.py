import os
import json
import cv2
import argparse
from tqdm import tqdm

def load_coco_annotations(coco_json_path):
    """
    COCO 형식 어노테이션 JSON 파일을 로드합니다.
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    return data

def visualize_annotations_on_image(image, annotations, cat_id_to_name):
    """
    단일 이미지에 대해 어노테이션 리스트를 기반으로 bbox와 라벨을 그립니다.
    """
    for ann in annotations:
        # COCO bbox 형식: [x, y, width, height]
        x, y, w, h = map(int, ann["bbox"])
        cat_id = ann["category_id"]
        cat_name = cat_id_to_name.get(cat_id, "N/A")
        # 녹색 bbox (두께 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 라벨 텍스트 (bbox 위에 표시)
        cv2.putText(image, cat_name, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def visualize_folder_images(image_folder, coco_data, save_folder):
    """
    이미지 폴더 내의 모든 이미지에 대해, COCO 어노테이션을 시각화하여 저장합니다.
    어노테이션이 없는 이미지는 건너뜁니다.
    """
    # COCO의 이미지 레코드를 파일명 기준으로 매핑
    file_to_image_record = {img["file_name"]: img for img in coco_data["images"]}
    # 이미지 id별 어노테이션 리스트 생성
    image_id_to_anns = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        image_id_to_anns.setdefault(image_id, []).append(ann)
        
    # 카테고리 id -> 이름 매핑
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    
    # 이미지 폴더 내의 이미지 파일 목록 (확장자 jpg, jpeg, png)
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 저장 폴더가 없으면 생성
    os.makedirs(save_folder, exist_ok=True)
    
    # tqdm 진행률 표시
    for file_name in tqdm(image_files, desc="Visualizing images"):
        # 어노테이션에 해당 이미지가 존재하지 않으면 건너뜀
        if file_name not in file_to_image_record:
            continue
        
        image_record = file_to_image_record[file_name]
        image_id = image_record["id"]
        
        image_path = os.path.join(image_folder, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: 이미지 '{image_path}'를 불러올 수 없습니다.")
            continue
        
        # 해당 이미지에 해당하는 어노테이션 추출 (없으면 빈 리스트)
        anns = image_id_to_anns.get(image_id, [])
        
        # 시각화 수행
        vis_image = visualize_annotations_on_image(image, anns, cat_id_to_name)
        
        # 결과 이미지 저장 (원본 파일명 동일)
        save_path = os.path.join(save_folder, file_name)
        cv2.imwrite(save_path, vis_image)
    
    print(f"\n시각화 완료! 결과 이미지가 '{save_folder}'에 저장되었습니다.")

def main():
    parser = argparse.ArgumentParser(
        description="COCO 형식 어노테이션을 기반으로 이미지 폴더 내 모든 이미지에 정답(ground truth)을 시각화합니다."
    )
    parser.add_argument("--image_folder", type=str, default="/dataset/bdd100k/images/100k/val")
    parser.add_argument("--ann", type=str, default="/dataset/bdd100k/coco_labels/det_20/bdd100k_det20_val_coco.json")
    parser.add_argument("--save_folder", type=str, default="/workspace/RT-DETR-WS/results/det20_val_img")
    args = parser.parse_args()
    
    # COCO 어노테이션 파일 로드
    coco_data = load_coco_annotations(args.ann)
    # 이미지 폴더 내 이미지에 대해 시각화 수행
    visualize_folder_images(args.image_folder, coco_data, args.save_folder)

if __name__ == '__main__':
    main()

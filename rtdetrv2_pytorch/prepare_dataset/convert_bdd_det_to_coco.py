"""
=== TRAIN 데이터셋 처리 시작 ===
Processing /dataset/bdd100k/labels/det_20/det_train.json: 100%|████████████████████████████████████████████████████████████████| 69863/69863 [00:03<00:00, 19516.29it/s]

[/dataset/bdd100k/labels/det_20/det_train.json] 총 이미지 개수: 69863
각 카테고리별 어노테이션 개수:
  pedestrian (id: 1): 92159개
  rider (id: 2): 4560개
  car (id: 3): 700703개
  truck (id: 4): 27892개
  bus (id: 5): 11977개
  train (id: 6): 128개
  motorcycle (id: 7): 3023개
  bicycle (id: 8): 7124개
  traffic light (id: 9): 187871개
  traffic sign (id: 10): 238270개

변환 완료! COCO 어노테이션 파일이 '/dataset/bdd100k/coco_labels/det_20/bdd100k_det20_train_coco.json'에 저장되었습니다.

=== VAL 데이터셋 처리 시작 ===
Processing /dataset/bdd100k/labels/det_20/det_val.json: 100%|██████████████████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 66937.72it/s]

[/dataset/bdd100k/labels/det_20/det_val.json] 총 이미지 개수: 10000
각 카테고리별 어노테이션 개수:
  pedestrian (id: 1): 13425개
  rider (id: 2): 658개
  car (id: 3): 102837개
  truck (id: 4): 4243개
  bus (id: 5): 1660개
  train (id: 6): 15개
  motorcycle (id: 7): 460개
  bicycle (id: 8): 1039개
  traffic light (id: 9): 26884개
  traffic sign (id: 10): 34724개

변환 완료! COCO 어노테이션 파일이 '/dataset/bdd100k/coco_labels/det_20/bdd100k_det20_val_coco.json'에 저장되었습니다.
"""

import json
from tqdm import tqdm

def convert_bdd100k_to_coco(bdd_json_path, output_path):
    """
    BDD100K JSON 파일을 COCO 어노테이션 포맷으로 변환하고,
    변환 후 이미지 개수 및 각 카테고리별 어노테이션 개수를 출력합니다.
    """
    # 입력 JSON 파일 읽기
    with open(bdd_json_path, 'r') as f:
        bdd_data = json.load(f)
    
    # BDD100K의 10개 클래스 (det_20이라는 이름이지만 실제 클래스는 10개)
    categories_list = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "rider"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "truck"},
        {"id": 5, "name": "bus"},
        {"id": 6, "name": "train"},
        {"id": 7, "name": "motorcycle"},
        {"id": 8, "name": "bicycle"},
        {"id": 9, "name": "traffic light"},
        {"id": 10, "name": "traffic sign"}
    ]
    
    # 카테고리 이름을 key로 하는 매핑 (소문자로 비교)
    cat_name_to_id = {cat["name"]: cat["id"] for cat in categories_list}

    # COCO 형식의 최종 출력 구조
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories_list
    }

    annotation_id = 1  # 어노테이션 id (1부터 순차 부여)
    image_id = 1       # 이미지 id (1부터 순차 부여)

    # tqdm을 이용하여 진행도 표시
    for item in tqdm(bdd_data, desc=f"Processing {bdd_json_path}"):
        # 이미지 정보 처리 (파일 이름, 해상도 등)
        image_entry = {
            "id": image_id,
            "file_name": item["name"],
            "width": 1280,   # BDD100K 기본 해상도 (필요시 변경)
            "height": 720
        }
        coco_output["images"].append(image_entry)

        # 각 이미지의 라벨(어노테이션) 처리
        for label in item.get("labels", []):
            # 카테고리 이름 (소문자, 공백 제거)
            category_name = label["category"].lower().strip()
            if category_name not in cat_name_to_id:
                continue
            cat_id = cat_name_to_id[category_name]

            # box2d 정보 추출 (x1, y1, x2, y2)
            box2d = label.get("box2d", None)
            if box2d is None:
                continue
            x1 = box2d["x1"]
            y1 = box2d["y1"]
            x2 = box2d["x2"]
            y2 = box2d["y2"]
            width = x2 - x1
            height = y2 - y1
            area = width * height

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, width, height],
                "area": area,
                "iscrowd": 0,
                "segmentation": []  # 폴리곤 정보가 없는 경우 빈 리스트
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

    # COCO 포맷 JSON 파일 저장
    with open(output_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    
    # 변환 후 이미지 개수 출력
    num_images = len(coco_output["images"])
    print(f"\n[{bdd_json_path}] 총 이미지 개수: {num_images}")
    
    # 각 카테고리별 어노테이션 개수 계산
    cat_counts = {cat["id"]: 0 for cat in categories_list}
    for ann in coco_output["annotations"]:
        cat_counts[ann["category_id"]] += 1
    
    print("각 카테고리별 어노테이션 개수:")
    for cat in categories_list:
        print(f"  {cat['name']} (id: {cat['id']}): {cat_counts[cat['id']]}개")
    
    print(f"\n변환 완료! COCO 어노테이션 파일이 '{output_path}'에 저장되었습니다.\n")
    return coco_output

if __name__ == '__main__':
    # train과 val 데이터셋에 대해 각각 변환 수행
    datasets = [
        ("train", "/dataset/bdd100k/labels/det_20/det_train.json", "/dataset/bdd100k/coco_labels/det_20/bdd100k_det20_train_coco.json"),
        ("val", "/dataset/bdd100k/labels/det_20/det_val.json", "/dataset/bdd100k/coco_labels/det_20/bdd100k_det20_val_coco.json")
    ]
    
    for split, input_json, output_json in datasets:
        print(f"=== {split.upper()} 데이터셋 처리 시작 ===")
        convert_bdd100k_to_coco(input_json, output_json)

# import os
# import requests
# import torch
# from flask import Flask, request, send_file
# import numpy as np
# import cv2
# import json
# from ultralytics import YOLO

# # Flask 앱 생성
# app = Flask(__name__)

# # 학습된 모델 파일의 경로
# best_model_path = "./train11/weights/best.pt"

# # 학습된 모델을 로드
# best_model = YOLO(best_model_path)

# # 이미지를 탐지하는 함수
# def detect_objects(image, model):
#     # 이미지를 YOLO 모델에 입력
#     results = model(image, stream=True)
#     # print("Input Image Shape:", image.shape)

#     detections = []
#     print("results: ", results)
#     for result in results:
#         print("Model Output:", result)
#         print("test1")
#         boxes = result.boxes  # Bbox 출력용 Boxes 객체
#         masks = result.masks  # Segmentation masks 출력용 Masks 객체
#         keypoints = result.keypoints  # Pose 출력용 Keypoints 객체
#         probs = result.probs  # Classification 출력용 Probs 객체
#         print("test2")
        
#         print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
#         print("11boxes: ", boxes)
#         print("11masks: ", masks)
#         print("11keypoints: ", keypoints)
#         print("11probs: ", probs)
#         print("11result: ", result)
        
#         # if boxes is not None and probs is not None:
#         #     class_id = int(boxes[4])  # 클래스 ID
#         #     class_name = best_model.names[class_id] if class_id in best_model.names else "unknown"
#         #     print("className: ", class_name)
#         #     # 검출 결과를 처리하고 표준 형식으로 변환
#         #     for box, prob in zip(boxes.xyxy, probs):
#         #         print("test3")
#         #         detections.append({
#         #             "box": box.tolist(),
#         #             "score": prob.item(),
#         #             "class": class_name  # 클래스 ID를 클래스 이름으로 매핑할 수 있습니다.
#         #         })

#     # return detections
    
#     return results


# # 이미지를 업로드하는 엔드포인트
# @app.route("/upload", methods=["POST"])
# def upload():
#     try:
#         # 이미지 파일을 업로드 받음
#         image_file = request.files["image"]

#         # 이미지를 배열로 변환
#         image = np.frombuffer(image_file.read(), np.uint8)
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#         # 이미지를 객체로 탐지
#         detections = detect_objects(image, best_model)
#         print("test4")

#         # 탐지된 객체를 JSON으로 변환
#         detections_json = json.dumps(detections)
#         print("test5")

#         # 탐지된 객체를 반환
#         return detections_json
#     except Exception as e:
#         print(f"요청 처리 중 오류 발생: {str(e)}")
#         return "오류", 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이미지 저장 방식ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
# import os
# import requests
# import torch
# from flask import Flask, request, send_file
# import numpy as np
# import cv2
# import json
# from ultralytics import YOLO

# # Flask 앱 생성
# app = Flask(__name__)

# # 이미지를 저장할 디렉토리 경로
# upload_folder = "./image_test/"
# os.makedirs(upload_folder, exist_ok=True)

# # 학습된 모델 파일의 경로
# best_model_path = "./train11/weights/best.pt"

# # 학습된 모델을 로드
# best_model = YOLO(best_model_path)

# # 이미지를 탐지하는 함수
# def detect_objects(image, model):
#     results = model(image, stream=True)
#     for result in results:
#         boxes = result.boxes
#         masks = result.masks
#         keypoints = result.keypoints
#         probs = result.probs
        
#         print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
#         print("11boxes: ", boxes)
#         print("11masks: ", masks)
#         print("11keypoints: ", keypoints)
#         print("11probs: ", probs)
#         print("11result: ", result)


#     return results

# # 이미지를 업로드하고 객체를 탐지하는 엔드포인트
# @app.route("/upload", methods=["POST"])
# def upload_and_detect():
#     try:
#         # 이미지 파일을 업로드 받음
#         image_file = request.files["image"]

#         # 이미지를 배열로 변환
#         image = np.frombuffer(image_file.read(), np.uint8)
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)

#         # 이미지를 저장
#         image_path = os.path.join(upload_folder, "uploaded_image.jpg")
#         cv2.imwrite(image_path, image)

#         # 이미지를 객체로 탐지
#         detections = detect_objects(image_path, best_model)

#         # 탐지된 객체를 JSON으로 변환
#         detections_json = json.dumps(detections)

#         # 탐지된 객체와 저장된 이미지의 경로를 반환
#         return {"detections": detections_json, "image_path": image_path}
#     except Exception as e:
#         print(f"요청 처리 중 오류 발생: {str(e)}")
#         return "오류", 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

from flask import Flask, request
import numpy as np
import cv2
import json
from ultralytics import YOLO

# Flask 앱 생성
app = Flask(__name__)

# 학습된 모델 파일의 경로
best_model_path = "./train7/weights/best.pt"

# 학습된 모델을 로드
best_model = YOLO(best_model_path)

# 이미지를 탐지하는 함수
def detect_objects(image, model):
    # 이미지를 YOLO 모델에 입력
    results = model.predict(image)
    print("000000000000000: ", results[0])
    print()
    print("boxes:::::::::::::: ", results[0].boxes)
    
    # 결과를 저장할 리스트 초기화
    detections_list = []

    i=0
    for result in results:
        print("iiiiiiiiiiiiiiiiiiiii: ", i)
        
        boxes = result.boxes  # Bbox 출력용 Boxes 객체
        probs = result.probs  # Classification 출력용 Probs 객체
        
        print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
        print("ㅡㅡㅡㅡㅡboxes: ", boxes)
        print("ㅡㅡㅡㅡㅡprobs: ", probs)
        print("ㅡㅡㅡㅡㅡresult: ", result)
        i=i+1
        
        # 딕셔너리로 변환
        detection_dict = {"boxes": boxes, "probs": probs}

        # 리스트에 추가
        detections_list.append(detection_dict)
    
    # 리스트를 JSON으로 직렬화
    detections_json = json.dumps(detections_list)

    return detections_json


# 이미지를 업로드하는 엔드포인트
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # 이미지 파일을 업로드 받음
        image_file = request.files["image"]

        # 이미지를 배열로 변환
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # 이미지를 객체로 탐지
        detections = detect_objects(image, best_model)
        print("test4")

        # 탐지된 객체를 JSON으로 변환
        detections_json = json.dumps(detections)
        print("test5")

        # 탐지된 객체를 반환
        return detections_json
    except Exception as e:
        print(f"요청 처리 중 오류 발생: {str(e)}")
        return "오류", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# from flask import Flask, request
# import numpy as np
# import cv2
# import json
# from ultralytics import YOLO

# # Flask 앱 생성
# app = Flask(__name__)

# # 학습된 모델 파일의 경로
# best_model_path = "./train7/weights/best.pt"

# # 학습된 모델을 로드
# best_model = YOLO(best_model_path)

# # 이미지를 탐지하는 함수
# def detect_objects(image, model):
#     # 이미지를 YOLO 모델에 입력
#     results = model.predict(image)
#     print("000000000000000: ", results[0])
#     print()
#     print("boxes:::::::::::::: ", results[0].boxes)
    
#     # probs 변수에 모델의 예측 결과를 저장
#     probs = results[0]
#     print("probs test: ", probs)
    
#     res_json = json.loads(results[0].tojson()) #이거다. 이 두 줄이 결과창이다! 이 안에 name, class 등 정보가 담겼다.
#     print("res_json: ", res_json)
    
#     # 결과를 저장할 리스트 초기화
#     detections_list = []

#     i=0
#     for result in results:
#         print("iiiiiiiiiiiiiiiiiiiii: ", i)
        
#         print("result.boxes.cls: ", result.boxes.cls)
#         print("result.boxes.conf: ", result.boxes.conf)
        
#         boxes = result.boxes  # Bbox 출력용 Boxes 객체
#         probs = result.probs  # Classification 출력용 Probs 객체
        
#         print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
#         print("ㅡㅡㅡㅡㅡboxes: ", boxes)
#         print("ㅡㅡㅡㅡㅡprobs: ", probs)
#         print("ㅡㅡㅡㅡㅡresult: ", result)
#         i=i+1
        
#         # 딕셔너리로 변환
#         detection_dict = {"probs": probs}

#         # 리스트에 추가
#         detections_list.append(detection_dict)
    
#     # 리스트를 JSON으로 직렬화
#     detections_json = json.dumps(detections_list)

#     return detections_json


# # 이미지를 업로드하는 엔드포인트
# @app.route("/upload", methods=["POST"])
# def upload():
#     try:
#         # 이미지 파일을 업로드 받음
#         image_file = request.files["image"]

#         # 이미지를 배열로 변환
#         # image = np.frombuffer(image_file.read(), np.uint8)
#         # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         image=cv2.imread(image_file)

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
    
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
from flask import Flask, request
import numpy as np
import cv2
import json
from ultralytics import YOLO

# Flask 앱 생성
app = Flask(__name__)

# 학습된 모델 파일의 경로
best_model_path = "./train8/weights/best.pt"

# 학습된 모델을 로드
best_model = YOLO(best_model_path)

# 이미지를 탐지하는 함수
def detect_objects(image, model):
    # 이미지를 YOLO 모델에 입력
    results = model.predict(image)
    
    res_json = json.loads(results[0].tojson()) #이거다. 이 두 줄이 결과창이다! 이 안에 name, class 등 정보가 담겼다.

    return res_json


# 이미지를 업로드하는 엔드포인트
@app.route("/upload", methods=["POST"])
def upload():
    try:
        # 이미지 파일을 업로드 받음
        image_file = request.files["image"]

        # 이미지를 배열로 변환
        image=cv2.imread(image_file)

        # 이미지를 객체로 탐지
        detections = detect_objects(image, best_model)
        
        print("최종 결과: ", detections)
        print("Class name: ", detections[0]['name'])
        print("Class number: ", detections[0]['class'])

        # 탐지된 객체를 반환
        return detections
    except Exception as e:
        print(f"요청 처리 중 오류 발생: {str(e)}")
        return "오류", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


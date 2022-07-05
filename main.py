from deepface import DeepFace

# DeepFace Models
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
# DeepFace Backends
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']


# 얼굴 두개를 비교 합니다.(1:1 비교값)
def deepface_recognition():
    result = DeepFace.verify(img1_path="./images/su1.png", img2_path="./images/su3.jpeg")
    print('[+] Face Verification :: ', result)

    # face verification
    model_choice_result = DeepFace.verify(img1_path="./images/su1.png", img2_path="./images/su3.jpeg",
                                          model_name=models[1])
    print('[+] Face Verification (+Model):: ', model_choice_result)


# 얼굴 한개(img1_path)를 기준으로 폴더를 비교합니다.(1:다 비교값)
def deepface_verification():
    result = DeepFace.find(img_path="dataset/jonghoon_lee.jpg", db_path="dataset", model_name=models[6])
    print('[+] Face recognition :: ', result)


# 얼굴 하나에 대한 감정, 나이, 성별, race를 분석해준다.
def deepface_analyze():
    result = DeepFace.analyze(img_path="./images/su1.png", actions=['emotion', 'age', 'gender', 'race'])
    print('[+] Face Analyze :: ', result)


# 스트림으로 사용자를 측정함.
def deepface_stream():
    result = DeepFace.stream(db_path="images", detector_backend='opencv', model_name="ArcFace", enable_face_analysis=False, time_threshold=1, frame_threshold=5)
    print('[+] Steam :: ', result)
    # print("")


if __name__ == '__main__':
    print('[+] init Start !!')
    # deepface_recognition()
    # deepface_verification()
    # deepface_analyze()
    deepface_stream()

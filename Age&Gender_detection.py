import cv2
import os

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

print("Current folder contents:")
for f in os.listdir(script_dir):
    if 'deploy' in f or 'caffemodel' in f:
        print(f"✅ FOUND: {f}")

# Webcam & FACE DETECTOR FIX
video_capture = cv2.VideoCapture(0)
# ✅ FIX: Use OpenCV's built-in cascade (100% works)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture.set(3, 640)
video_capture.set(4, 480)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def load_caffe_models():
    age_proto = os.path.join(script_dir, 'deploy_age.prototxt')
    age_model = os.path.join(script_dir, 'age_net.caffemodel')
    gender_proto = os.path.join(script_dir, 'deploy_gender.prototxt')
    gender_model = os.path.join(script_dir, 'gender_net.caffemodel')
    
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
    return age_net, gender_net

def video_detector(age_net, gender_net):
    while True:
        ret, frame = video_capture.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Face with padding
            padding = 20
            face_img = frame[max(0,y-padding):min(frame.shape[0],y+h+padding),
                           max(0,x-padding):min(frame.shape[1],x+w+padding)].copy()
            
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Predictions
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Age & Gender Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)

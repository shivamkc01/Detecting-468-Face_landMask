import cv2
import mediapipe as mp
import time

VIDEO_PATH = "/Users/rishav00a/PycharmProjects/Detect_Face_landmarks/videos/Demo.mp4"
camera = cv2.VideoCapture(VIDEO_PATH)
pTime = 0

mpDraw = mp.solutions.drawing_utils  # This will helps us to draw mask over faces
mpFaceMask = mp.solutions.face_mesh
faceMask = mpFaceMask.FaceMesh(max_num_faces=2)
"""
  def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
    Initializes a MediaPipe FaceMesh object.

    Args:
      static_image_mode: Whether to treat the input images as a batch of static
        and possibly unrelated images, or a video stream. See details in
        https://solutions.mediapipe.dev/face_mesh#static_image_mode.
      max_num_faces: Maximum number of faces to detect. See details in
        https://solutions.mediapipe.dev/face_mesh#max_num_faces.
      min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
        detection to be considered successful. See details in
        https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.
      min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
        face landmarks to be considered tracked successfully. See details in
        https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.

"""
drawSpac = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


while True:
    success, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMask.process(imgRGB)
    if results.multi_face_landmarks:
        # we are getting facelandmash of one face
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,
                                  faceLms,
                                  mpFaceMask.FACEMESH_CONTOURS,
                                  drawSpac,
                                  drawSpac
                                  )
            for id, lm in enumerate(faceLms.landmark):
                # print(lm) -> getting landmark x, y & z position values
                # Converting landmark positions values into pixels
                img_h, img_w, img_c = img.shape
                # first getting x & y values
                x, y = int(lm.x*img_w), int(lm.y*img_h)
                print(id, x, y)



    cTime = time.time()
    fps = 1/(cTime-pTime)   # cTime -> currentTime, pTime -> PreviousTime
    pTime = cTime
    cv2.putText(img, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
import cv2
import mediapipe as mp
import time

VIDEO_PATH = "/Users/rishav00a/PycharmProjects/Detect_Face_landmarks/videos/Demo.mp4"


class faceMaskDetector():
    def __init__(self,
                 staticMode=False,
                 maxFaces=2,
                 minDetectionCon=0.5,
                 maxDetectionCon=0.5,
                 thickness=1,
                 circle_radius=2):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.maxDetectionCon = maxDetectionCon
        self.thickness = thickness
        self.circle_radius = circle_radius

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMask = mp.solutions.face_mesh
        self.FaceMask = self.mpFaceMask.FaceMesh(self.staticMode,
                                                 self.maxFaces,
                                                 self.minDetectionCon,
                                                 self.maxDetectionCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=self.thickness,
                                                circle_radius=self.circle_radius)

    def findFaces(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.FaceMask.process(self.imgRGB)
        faces = []  # Total no of faces stored in faces list
        if self.results.multi_face_landmarks:
            # we are getting facelandmash of one face
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                               faceLms,
                                               self.mpFaceMask.FACEMESH_CONTOURS,
                                               self.drawSpec,
                                               self.drawSpec
                                               )

                face = []   # counting the no of faces
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm) -> getting landmark x, y & z position values
                    # Converting landmark positions values into pixels
                    img_h, img_w, img_c = img.shape
                    # first getting x & y values
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                0.7, (0, 255, 0), 1)
                    #print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    camera = cv2.VideoCapture(VIDEO_PATH)
    pTime = 0
    detector = faceMaskDetector()
    while True:
        success, img = camera.read()
        img, faces = detector.findFaces(img)
        if len(faces) != 0:
            print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)  # cTime -> currentTime, pTime -> PreviousTime
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("FaceMash Detector", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

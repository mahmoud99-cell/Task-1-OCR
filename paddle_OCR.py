
# pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# pip install paddlehub -i https://mirror.baidu.com/pypi/simple
# hub install chinese_text_detection_db_server


# Made By: Mahmoud Nabil Hessein

import paddlehub as hub
import cv2
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images     



# ocr = hub.Module(name="chinese_ocr_db_crnn_server")
ocr=hub.Module.init_with_name(name="chinese_ocr_db_crnn_server")


# folder_imgs= load_images_from_folder("path/to/folder/images") #enter path here !!!
# for image in folder_imgs:
#     roi = cv2.selectROI(img)
#     #print rectangle points of selected roi
#     print(roi)

#     #Crop selected roi from raw image
#     roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
#     #show cropped image
#     cv2.imshow("ROI", roi_cropped)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     result = ocr.recognize_text(images=[roi_cropped])

#     finaldata=result[0]["data"]
#     s=""
#     print("Number of text detected= "+str(len(finaldata)))
#     print('\n')
#     print(finaldata)
#     for textINFO in finaldata:
#         print(textINFO)
#         print("\n")
#         s=s+textINFO["text"]
#         print("\n")

#     print("\n Final Text: "+s)


img=cv2.imread('test.jpg')
roi = cv2.selectROI(img)

#print rectangle points of selected roi
print(roi)

#Crop selected roi from raw image
roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
#show cropped image
cv2.imshow("ROI", roi_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = ocr.recognize_text(images=[roi_cropped])

finaldata=result[0]["data"]
s=""
print("Number of text detected= "+str(len(finaldata)))
print('\n')
print(finaldata)
for textINFO in finaldata:
    print(textINFO)
    print("\n")
    s=s+textINFO["text"]
    print("\n")

print("\n Final Text: "+s)



"""


Final Text: This is a lot of 12 point text to test theocr code and see if it works on all typesof file format.
The quick brown dog jumped over thelazy fox.
The quick brown dog jumpedover the lazy fox.
The quick brown dogjumped over the lazy fox. 
The quickbrown dog jumped over the lazy fox.
"""
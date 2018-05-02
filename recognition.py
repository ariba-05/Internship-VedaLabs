import time
import cv2
import numpy as np
import dlib
from skimage import io


def violaJones(resolution,changeResol = True): 
    totalTime = 0 
    #change path to the xml file in your system     
    face_cascade = cv2.CascadeClassifier('/home/ariba/OpenCV/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')

    #there's a folder 'output-detection' in the main directory in which create 10 .txt files in advance by the name 'fold-xx-out.txt'(0<=xx<=10)
    for ii in range(1,11):
        print('Fold {} ...'.format(ii))
        path = 'FDDB-folds/FDDB-fold-'+str(ii).zfill(2)+'.txt'
        inFile = open(path,'r') #open fold-xx.txt file which contains list of images in that fold
	    
        listOfFiles = inFile.readlines() #put these paths to images in a single list
        inFile.close()
	    
        path = 'output-detection/fold-'+str(ii).zfill(2)+'-out.txt'
        outFile = open(path,'w') #open detection file 
        #reading image one by one from the list and applying the detection algo  
        for impath in listOfFiles:
            impath = 'original-pics/'+impath[:-1]+'.jpg'
            img = cv2.imread(impath)
            if changeResol == True:
                img = resizeImg(img,resolution)
            img = cv2.UMat(img)
            #print(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            start = time.time()
            faces = face_cascade.detectMultiScale2(gray, 1.3, 5)
            totalTime += time.time() - start
            #if no faces are found in an image
            if not faces:
                outFile.write(impath[14:-4]+str(len(faces[0])))
                continue
	
            numDetections = faces[1]

            string=impath[14:-4]+'\n'+str(len(faces[0]))
            i=0
            #output-specification jhamela 
            for (x,y,w,h) in faces[0]:
                string = string + '\n' + str(x) + ' ' +str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + str(numDetections[i][0])
                i+=1
            string += '\n'
            outFile.write(string)
        outFile.close()
    avgTime = totalTime/2845
    outW, outH = resolution
    if changeResol == True:
        print('Avg time to detect faces/frame resolution of ',outH,'p through OpenCV Haar cascades is : ',avgTime,'s')
    else:
        print('Avg time to detect faces/frame without changing resolution through OpenCV Haar cascades is :',avgTime,'s')

def dlibHOG(resolution,changeResol = True): 
    totalTime = 0 
    
    detector = dlib.get_frontal_face_detector()
    #there's a folder 'output-detection' in the main directory in which create 10 .txt files in advance by the name 'fold-xx-out.txt'(0<=xx<=10)
    for ii in range(1,11):
        print('Fold {} ...'.format(ii))
        path = 'FDDB-folds/FDDB-fold-'+str(ii).zfill(2)+'.txt'
        inFile = open(path,'r') #open fold-xx.txt file which contains list of images in that fold
	    
        listOfFiles = inFile.readlines() #put these paths to images in a single list
        inFile.close()
	    
        path = 'output-detection/fold-'+str(ii).zfill(2)+'-out.txt'
        outFile = open(path,'w') #open detection file 
        #reading image one by one from the list and applying the detection algo  
        for impath in listOfFiles:
            impath = 'original-pics/'+impath[:-1]+'.jpg'
            img = cv2.imread(impath)
            if changeResol == True:
                img = resizeImg(img,resolution)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            start = time.time()
            dets, scores, idx = detector.run(gray,1,-1)
            totalTime += time.time() - start
            #if no faces are found in an image
            if not dets:
                outFile.write(impath[14:-4]+'\n'+str(0)+'\n')
                continue
          
            string=impath[14:-4]+'\n'+str(sum(x>0 for x in scores))#using only the positive score detections
            i=0
            #output-specification jhamela 
            for i,d in enumerate(dets):
                if scores[i]>0:
                    string = string + '\n' + str(d.left()) + ' ' +str(d.top()) + ' ' + str(d.right()-d.left()) + ' ' + str(d.bottom()-d.top()) + ' ' + str(scores[i])
                    i+=1
                    continue
                else:
                    break
            string += '\n'
            outFile.write(string)
        outFile.close()
    avgTime = totalTime/2845
    outW, outH = resolution
    if changeResol == True:
        print('Avg time to detect faces/frame resolution of ',outH,'p through Dlib HOG is : ',avgTime,'s')
    else:
        print('Avg time to detect faces/frame without changing resolution through Dlib HOG is :',avgTime,'s')


def dlibCNN(resolution,changeResol = True): 
    totalTime = 0 
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
    #there's a folder 'output-detection' in the main directory in which create 10 .txt files in advance by the name 'fold-xx-out.txt'(0<=xx<=10)
    for ii in range(1,11):
        print('Fold {} ...'.format(ii))
        path = 'FDDB-folds/FDDB-fold-'+str(ii).zfill(2)+'.txt'
        inFile = open(path,'r') #open fold-xx.txt file which contains list of images in that fold
	    
        listOfFiles = inFile.readlines() #put these paths to images in a single list
        inFile.close()
	    
        path = 'output-detection/fold-'+str(ii).zfill(2)+'-out.txt'
        outFile = open(path,'w') #open detection file 
        #reading image one by one from the list and applying the detection algo  
        for impath in listOfFiles:
            impath = 'original-pics/'+impath[:-1]+'.jpg'
            img = cv2.imread(impath)
            if changeResol == True:
                img = resizeImg(img,resolution)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            start = time.time()
            dets = cnn_face_detector(gray, 1)
            totalTime += time.time() - start
            #if no faces are found in an image
            if not dets:
                outFile.write(impath[14:-4]+'\n'+str(0)+'\n')
                continue
          
            string=impath[14:-4]+'\n'+str(len(dets))#using only the positive score detections
            i=0

            #output-specification jhamela 

            for i,d in enumerate(dets):

                string = string + '\n' + str(d.rect.left()) + ' ' +str(d.rect.top()) + ' ' + str(d.rect.right()-d.rect.left()) + ' ' + str(d.rect.bottom()-d.rect.top()) + ' ' + str(d.confidence)

            string += '\n'

            outFile.write(string)

        outFile.close()

    avgTime = totalTime/2845

    outW, outH = resolution

    if changeResol == True:

        print('Avg time to detect faces/frame resolution of ',outH,'p through Dlib CNN is : ',avgTime,'s')

    else:

        print('Avg time to detect faces/frame without changing resolution through Dlib CNN is :',avgTime,'s')





def resizeImg(img, size, keepAspect = True, padding = True):
    """ Resize the image to given size.
    img         -- input source image
    size        -- (w,h) of desired resized image
    keepAspect  -- to preserve aspect ratio during resize 
    padding     -- to add black padding when target aspect is different 
    """
    dtype = img.dtype
    outW, outH = size

    if len(img.shape)>2:
        h, w, d = img.shape[:3]
        if padding:
            outimg = np.zeros((outH, outW, d), dtype=dtype)
    else:
        h, w = img.shape[:2]
        if padding:
            outimg = np.zeros((outH, outW), dtype=dtype)

    if keepAspect:
        aspect = float(w)/h
        if int(outH*aspect) < outW:   #output image is wider so limiting factor is height
            out = cv2.resize(img, (int(outH*aspect), outH))
            if padding:
                outimg[:, int((outW-int(outH*aspect))/2):int((outW+int(outH*aspect))/2), ] = out
                out = outimg
        else:
            out = cv2.resize(img, (outW, int(outW/aspect)))
            if padding:
                outimg[int((outH-int(outW/aspect))/2):int((outH+int(outW/aspect))/2), ] = out
                out = outimg
    else:
        out = cv2.resize(img, size)
    return out


if __name__ == '__main__':
    '''
     specify:
     violaJones((width,height)) if you wish to change resolution
     or violaJones((anything),False) if you do not

    '''
    violaJones((480,360),False)
    #dlibHOG((480, 360))
    #dlibCNN((480,360),False)



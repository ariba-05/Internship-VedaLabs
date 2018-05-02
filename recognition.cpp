#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif


#include <iostream>
#include <iomanip>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;


static GpuMat convertAndResize(const GpuMat& src, GpuMat& gray)
{

    if (src.channels() == 3)
    {
        cv::cuda::cvtColor( src, gray, COLOR_BGR2GRAY );
    }
    else
    {
        gray = src;
    }
    //cout<<gray.rows;
    int w = gray.cols;
    int h = gray.rows;

    float aspect = float(w)/h;

    //MAKE CHANGES HERE
    float outW = 480;
    float outH = 360;
    
    Size sz1(int(outH*aspect), outH), sz2(outW, int(outW/aspect));
    //cout<<aspect<<endl;
    GpuMat outimg_gpu(outW, outH, CV_8UC1), out, mask1(outW, outH, CV_8UC3), mask2(outW, outH, CV_8UC1);
    outimg_gpu.setTo(Scalar::all(0));

    if(int(outH*aspect) < outW){     //output image is wider so limiting factor is height
    	cv::cuda::resize(gray, out, sz1);
	cout<<"im here";
	mask1.colRange(int((outW-int(outH*aspect))/2),int((outW+int(outH*aspect))/2)).setTo(Scalar(0));
	out.copyTo(outimg_gpu,mask1); //outimg_gpu contains out at places where mask is set
    }
    else{
	cv::cuda::resize(gray, out, sz2);
	mask2.rowRange(int((outH-int(outW/aspect))/2),int((outH+int(outW/aspect))/2)).setTo(Scalar(0));
	out.copyTo(outimg_gpu,mask2);
    }

    
    return outimg_gpu;
}


static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.8;
    int fontThickness = 2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, Scalar(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}


static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorRed = Scalar(255,0,0);
    Scalar fontColorNV  = Scalar(118,185,0);

    ostringstream ss;
    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorRed, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "], " <<
        (bGpu ? "GPU, " : "CPU, ") <<
        (bLargestFace ? "OneFace, " : "MultiFace, ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");
    matPrint(canvas, 1, fontColorRed, ss.str());

    // by Anatoly. MacOS fix. ostringstream(const string&) is a private
    // matPrint(canvas, 2, fontColorNV, ostringstream("Space - switch GPU / CPU"));
    if (bHelp)
    {
        matPrint(canvas, 2, fontColorNV, "Space - switch GPU / CPU");
        matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}



int main(){

	string cascadeName = "haarcascade_frontalface_alt.xml";
	string inputName = "/home/ariba/img_348.jpg";
	
	Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(cascadeName);

	cv::CascadeClassifier cascade_cpu;
	Mat image = imread(inputName);
	//cout<<image;

	vector<Rect> faces;
	Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
	GpuMat frame_gpu, gray_gpu, facesBuf_gpu, resized_gpu;

	/* parameters */
	bool useGPU = true;
	double scaleFactor = 1.0;
	bool findLargestObject = false;
	bool filterRects = true;
	bool helpScreen = false;

	for(int g=0;g<10;g++){
		
		//image.copyTo(frame_cpu);
                frame_gpu.upload(image);
		
		resized_gpu = convertAndResize(frame_gpu, gray_gpu);
        	
		TickMeter tm;
		tm.start();

		cascade_gpu->setFindLargestObject(findLargestObject);
		cascade_gpu->setScaleFactor(1.2);
		cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);

		cascade_gpu->detectMultiScale(resized_gpu, facesBuf_gpu);
		cascade_gpu->convert(facesBuf_gpu, faces);
			
		for (size_t i = 0; i < faces.size(); ++i)
		{
		    rectangle(resized_cpu, faces[i], Scalar(255));
		}

		tm.stop();
		double detectionTime = tm.getTimeMilli();
		double fps = 1000 / detectionTime;

		cout << setfill(' ') << setprecision(2);
		cout << setw(6) << fixed << fps << " FPS, " << faces.size() << " det";
		if ((filterRects || findLargestObject) && !faces.empty())
		{
		    for (size_t i = 0; i < faces.size(); ++i)
		    {
		        cout << ", [" << setw(4) << faces[i].x
		             << ", " << setw(4) << faces[i].y
		             << ", " << setw(4) << faces[i].width
		             << ", " << setw(4) << faces[i].height << "]";
		    }
		}
		cout << endl;

		cv::cvtColor(resized_cpu, frameDisp, COLOR_GRAY2BGR);
		displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
		imshow("result", frameDisp);

	}

return 0;
}









// self.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include<opencv2/world.hpp>
#include<opencv2/objdetect.hpp>
using namespace std;
using namespace cv;

/** Function Headers */
void mydetectAndDisplay(Mat & frame, Point &left, Point &right,int &radii, bool &flag);
/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main(void)
{
	Mat frame;//调用摄像头用
	Mat gray;//获得灰度图像时使用
	Mat src, dst, mask, result;//融合使用
	Point left, right;//大小变换使用
	bool flag = 0;
	int radii = 0;
	double dis = 0;

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };

	cout << "--------------------------------" << endl;
	cout << "please choose 1 or 2 to select:" << endl;
	cout << "1.get photo from camera;" << endl;
	cout << "2.use photo which we have;" << endl;
	cout << "--------------------------------" << endl;
	int input;
	cin >> input;

	if (input == 1)
	{
		VideoCapture capture;
		
		//-- 2. Read the video stream
		capture.open(0);
		if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

		while (capture.read(frame))
		{
			if (frame.empty())
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			imshow(window_name, frame);
			if (waitKey(10) == 27)
			{
				dst = frame;
				break;
			}
		}
	}
	else
	{
		frame = imread("C:\\Users\\lyuanbowen\\Documents\\Visual Studio 2013\\Projects\\self\\self\\图片源\\sor2.jpg");
		if (!frame.data)
		{
			printf("No data!--Exiting the program \n");
			return -1;
		}
	}
	mydetectAndDisplay( frame ,left,right,radii,flag);
	if (flag = 0)
	{ cout << "photo is illegal" << endl; return 0; }
	dst = frame;
	src = imread("C:\\Users\\lyuanbowen\\Documents\\Visual Studio 2013\\Projects\\self\\self\\图片源\\glasses_c2.jpg");
	if (!src.data) // Check for invalid input  
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	dis =2.5 * abs(left.x - right.x);//预估计人脸宽度
	double scale = dis / double(src.cols);//设置缩放比例
	//cout << "dis=" << dis << endl;
	//cout << "scale=" << scale << endl;
	//cout << "mask.cols=" << mask.cols << endl;
	Size dsize = Size(src.cols*scale, src.rows*scale);
	Mat image2 = Mat(dsize, CV_32S);
	resize(src, image2, dsize);
	src = image2;
	imshow("image", src);
	//imshow("srcchenge",image2);
	
	mask = imread("C:\\Users\\lyuanbowen\\Documents\\Visual Studio 2013\\Projects\\self\\self\\图片源\\maskc23.png");
	if (!mask.data) // Check for invalid input  
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	Size dsize2 = Size(mask.cols*scale, mask.rows*scale);
	Mat image3 = Mat(dsize2, CV_32S);
	resize(mask, image3, dsize2);
	mask = image3;

	//cvtColor(src,gray,CV_BGR2GRAY);//获得灰度图像
	//int threshold_value = 100;//设置阀值
	//int max_BINARY_value = 255;
	//mask = Mat::zeros(src.size(), CV_8UC1);
	//threshold(gray, mask, threshold_value, max_BINARY_value, CV_THRESH_BINARY_INV);//二值化处理
	//cout << "mask_ch=" << mask.channels() << endl;
	
	imshow("mask", mask);
	
	Point p;
	p.x =  (left.x + right.x) / 2;
	p.y =  (left.y + right.y) / 2 + radii/3;
	//cout << "p.x=" << p.x << endl;
	//cout << "p.y=" << p.y << endl;
	
	seamlessClone(src, dst, mask, p, result, 1);//融合
	imwrite("..\\self\\图片源\\result.jpg", result);
	imshow("result", result);

	Mat logo=imread("");
	// define image ROI  
	//cv::Mat imageROI;
	//imageROI = dst(cv::Rect(385, 270, logo.cols, logo.rows));
	// add logo to image   
	//cv::addWeighted(imageROI, 1.0, logo, 0.3, 0., imageROI);


	cout << "--------------------------------" << endl;
	cout << "push any key to end." << endl;
	cout << "--------------------------------" << endl;
	waitKey(0);
	/*
	while (true)
	{
		imshow(window_name, frame);
		if (waitKey(10) == 27)
		{
			break;
		}
	}
	*/
	return 0;
		
	
	
}

/** @function detectAndDisplay */
void mydetectAndDisplay(Mat & frame,Point &left,Point &right,int &radii,bool &flag)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);//转换为灰度图像
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		//ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		if (eyes.size() == (size_t)2){ flag = 1; }
		else { flag = 0; return; }
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			if (j == 0) left = eye_center;
			else right = eye_center;
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			radii = radius;
			//circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
	}
}
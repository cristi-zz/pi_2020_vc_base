// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>
using namespace std;



void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("opened image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat_<Vec3b> src = imread(fname, IMREAD_COLOR);

		int height = src.rows;
		int width = src.cols;

		Mat_<uchar> dst(height, width);

		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("original image",src);
		imshow("gray image",dst);
		waitKey();
	}
}



void createSignature()
{
	ifstream file("USERS/USER1/SIGN_FOR_USER1_USER2_2.csv");
	vector<vector<double>> coordinates;

	// Discard the first row (header)
	string header;
	getline(file, header);

	// Read each line and extract the x and y coordinates
	string line;
	while (getline(file, line)) {
		vector<double> row;
		stringstream ss(line);
		string value;

		while (getline(ss, value, ',')) {
			row.push_back(stod(value));
		}

		coordinates.push_back({ row[0],row[1] });
	}

	// Create a new image or load an existing one
	Mat img = Mat::zeros(1000, 1500, CV_8UC1);

	// Convert the coordinates into OpenCV format
	vector<Point> points;
	for (const auto& row : coordinates) {
		int x = static_cast<int>(row[0]);
		int y = static_cast<int>(row[1]);
		points.push_back(Point(x, y));

	}
	// Draw the signature using polylines
	polylines(img, points, false, Scalar(255, 255, 255), 2, LINE_AA);

	// Display the image
	imshow("Signature", img);
	waitKey(0);
}
int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Basic image opening...\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Color to Gray\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testColor2Gray();
				break;
			case 4:
				createSignature();
				break;
		}
	}
	while (op!=0);
	return 0;
}
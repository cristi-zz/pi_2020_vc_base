// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "gnuplot-iostream.h" // Gnuplot library


using namespace std;
using namespace cv;




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



void createSignatureGenuine(int num)
{
	for (int i = 46; i <= 65;i++) {
		string filename1 = "USERS/USER"+ to_string(num)+ "/" + to_string(num)+ " (" + to_string(i) + ").csv";
		ifstream file(filename1);

		vector<vector<double>> coordinates;
		string header;
		getline(file, header); // skip the first line

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

		Mat img = Mat::zeros(1000, 1500, CV_8UC1);
		vector<Point> points;
		for (const auto& row : coordinates) {
			int x = static_cast<int>(row[0]);
			int y = static_cast<int>(row[1]);
			points.push_back(Point(x, y));
		}
		polylines(img, points, false, Scalar(255, 255, 255), 2, LINE_AA);
		imwrite("signature_" + to_string(i) + ".jpg", img);
	}

	waitKey(0);
}

void createSignatureForged(int num)
{
	for (int i = 1; i <= 20; i++) {
		string filename1 = "USERS/USER" + to_string(num) + "/" + to_string(num) + " (" + to_string(i) + ").csv";
		ifstream file(filename1);

		vector<vector<double>> coordinates;
		string header;
		getline(file, header); // skip the first line

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

		Mat img = Mat::zeros(1000, 1500, CV_8UC1);
		vector<Point> points;
		for (const auto& row : coordinates) {
			int x = static_cast<int>(row[0]);
			int y = static_cast<int>(row[1]);
			points.push_back(Point(x, y));
		}
		polylines(img, points, false, Scalar(255, 255, 255), 2, LINE_AA);
		imwrite("signature_" + to_string(i) + ".jpg", img);
	}

	waitKey(0);
}

void createTestSignature(int num,int user)
{
		string filename = "USERS/USER"+to_string(user)+ "/" +to_string(user)+ " ("+to_string(num) + ").csv";
		ifstream file(filename);

		vector<vector<double>> coordinates;
		string header;
		getline(file, header); // skip the first line

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

		Mat img = Mat::zeros(1000, 1500, CV_8UC1);
		vector<Point> points;
		for (const auto& row : coordinates) {
			int x = static_cast<int>(row[0]);
			int y = static_cast<int>(row[1]);
			points.push_back(Point(x, y));
		}
		polylines(img, points, false, Scalar(255, 255, 255), 2, LINE_AA);
		
		imwrite("signature.jpg", img);
		waitKey(0);
	
}

vector<Point2f> extractFeatures(Mat& signature) {
	// Apply Gaussian filter to remove noise
	Mat signature_filtered;
	GaussianBlur(signature, signature_filtered, Size(5, 5), 0, 0);

	// Apply thresholding to convert to binary image
	Mat signature_binary;
	threshold(signature_filtered, signature_binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	normalize(signature_binary, signature_binary, 0, 1, NORM_MINMAX, CV_32F);


	// Apply Canny edge detection to extract boundary
	Mat signature_edges = signature.clone();

	// Find the center of the signature image
	Moments m = moments(signature_edges, true);
	Point center(m.m10 / m.m00, m.m01 / m.m00);

	// Remove lines longer than threshold
	double threshold = 150;
	// Find lines in the image
	vector<Vec4i> lines;
	HoughLinesP(signature_edges, lines, 1, CV_PI / 180, threshold, 0, 0);

	// Draw lines over the image
	Mat signature_cleaned = signature_edges.clone();
	vector<vector<Point>> polylines;
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		Point pt1(line[0], line[1]);
		Point pt2(line[2], line[3]);
		polylines.push_back({ pt1, pt2 });
	}

	cv::polylines(signature_cleaned, polylines, true, Scalar(0), 3, LINE_AA);

	circle(signature_cleaned, center, 5, Scalar(255, 0, 255), -1);

	int midx = center.x;
	Mat left_half = signature_cleaned(Rect(0, 0, midx, signature_cleaned.rows));
	// Find center of the left half
	Moments left_m = moments(left_half, true);
	Point left_center(left_m.m10 / left_m.m00, left_m.m01 / left_m.m00);
	circle(signature_cleaned, left_center, 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned, Point(0, left_center.y), Point(center.x, left_center.y), Scalar(255, 255, 255), 2);

	// Find center of the right half
	Mat right_half = signature_cleaned(Rect(midx, 0, signature_cleaned.cols - midx, signature_cleaned.rows));
	Moments right_m = moments(right_half, true);
	Point right_center(midx + right_m.m10 / right_m.m00, right_m.m01 / right_m.m00);
	circle(signature_cleaned, right_center, 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned, Point(signature_cleaned.cols, right_center.y), Point(center.x, right_center.y), Scalar(255, 255, 255), 2);


	line(signature_cleaned, Point(center.x, 0), Point(center.x, signature_filtered.rows), Scalar(255, 255, 255), 2);

	Mat top_left = signature_cleaned(Rect(0, 0, midx, left_center.y));
	Moments top_left_m = moments(top_left, true);
	Point top_left_center(top_left_m.m10 / top_left_m.m00, top_left_m.m01 / top_left_m.m00);
	circle(signature_cleaned, top_left_center, 5, Scalar(255, 0, 255), -1);

	Mat top_right = signature_cleaned(Rect(midx, 0, signature_cleaned.cols - midx, top_left.rows));
	Moments top_right_m = moments(top_right, true);
	Point top_right_center(midx + top_right_m.m10 / top_right_m.m00, top_right_m.m01 / top_right_m.m00);
	circle(signature_cleaned, top_right_center, 5, Scalar(255, 0, 255), -1);

	// Extract bottom left quarter
	Mat bottom_left = signature_cleaned(Rect(0, left_center.y, midx, signature_cleaned.rows - left_center.y));
	Moments bottom_left_m = moments(bottom_left, true);
	Point bottom_left_center(bottom_left_m.m10 / bottom_left_m.m00, left_center.y + bottom_left_m.m01 / bottom_left_m.m00);
	circle(signature_cleaned, bottom_left_center, 5, Scalar(255, 0, 255), -1);

	// Extract bottom right quarter
	Mat bottom_right = signature_cleaned(Rect(midx, left_center.y, signature_cleaned.cols - midx, signature_cleaned.rows - left_center.y));
	Moments bottom_right_m = moments(bottom_right, true);
	Point bottom_right_center(midx + bottom_right_m.m10 / bottom_right_m.m00, left_center.y + bottom_right_m.m01 / bottom_right_m.m00);
	circle(signature_cleaned, bottom_right_center, 5, Scalar(255, 0, 255), -1);

	Mat signature_cleaned_2 = signature_cleaned.clone();
	Point center_1(m.m10 / m.m00, m.m01 / m.m00);
	circle(signature_cleaned_2, center_1, 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned_2, Point(0, center_1.y), Point(signature_filtered.cols, center_1.y), Scalar(255, 255, 255), 2);
	midx = center_1.x;

	Mat top_half = signature_cleaned_2(Rect(0, 0, signature_cleaned_2.cols, center.y));
	Moments top_m = moments(top_half, true);
	Point top_center(top_m.m10 / top_m.m00, top_m.m01 / top_m.m00);
	circle(top_half, top_center, 5, Scalar(255, 0, 255), -1);
	line(top_half, Point(top_center.x, 0), Point(top_center.x, center.y), Scalar(255, 255, 255), 2);

	int midy = center.y;
	Mat bottom_half = signature_cleaned_2(Rect(0, midy, signature_cleaned_2.cols, signature_cleaned_2.rows - midy));
	Moments bottom_m = moments(bottom_half, true);
	Point bottom_center(bottom_m.m10 / bottom_m.m00, bottom_m.m01 / bottom_m.m00);
	circle(signature_cleaned_2, bottom_center + Point(0, midy), 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned_2, Point(bottom_center.x, midy), Point(bottom_center.x, signature_filtered.rows), Scalar(255, 255, 255), 2);

	Rect left_roi_top(0, 0, top_half.cols / 2, top_half.rows);
	Mat left_half_top = top_half(left_roi_top);

	// Compute moments and mass center for left section
	Moments left_moments_top = moments(left_half_top, true);
	Point left_center_top(left_moments_top.m10 / left_moments_top.m00, left_moments_top.m01 / left_moments_top.m00);
	circle(top_half, left_center_top, 5, Scalar(255, 255, 255), -1);

	Rect right_roi_top(top_half.cols / 2, 0, top_half.cols / 2, top_half.rows);
	Mat right_half_top = top_half(right_roi_top);

	// Compute moments and mass center for right section
	Moments right_moments_top = moments(right_half_top, true);
	Point right_center_top(right_moments_top.m10 / right_moments_top.m00 + top_half.cols / 2, right_moments_top.m01 / right_moments_top.m00);
	circle(top_half, right_center_top, 5, Scalar(255, 0, 255), -1);



	Rect left_roi_bottom(0, 0, bottom_half.cols / 2, bottom_half.rows);
	Mat left_half_bottom = bottom_half(left_roi_bottom);
	Moments left_moments_bottom = moments(left_half_bottom, true);
	Point left_center_bottom(left_moments_bottom.m10 / left_moments_bottom.m00, left_moments_bottom.m01 / left_moments_bottom.m00);
	circle(bottom_half, left_center_bottom, 5, Scalar(255, 255, 255), -1);

	Rect right_roi_bottom(bottom_half.cols / 2, 0, bottom_half.cols / 2, top_half.rows);

	Mat right_half_bottom = bottom_half(right_roi_bottom);
	Moments right_moments_bottom = moments(right_half_bottom, true);
	Point right_center_bottom(right_moments_bottom.m10 / right_moments_bottom.m00 + bottom_half.cols / 2, right_moments_bottom.m01 / right_moments_bottom.m00);
	circle(bottom_half, right_center_bottom, 5, Scalar(255, 0, 255), -1);
	std::vector<cv::Point2f> signature1;
	signature1.push_back(left_center);
	signature1.push_back(right_center);
	signature1.push_back(top_left_center);
	signature1.push_back(top_right_center);
	signature1.push_back(bottom_left_center);
	signature1.push_back(bottom_right_center);
	signature1.push_back(top_center);
	signature1.push_back(bottom_center);
	signature1.push_back(left_center_top);
	signature1.push_back(right_center_top);
	signature1.push_back(left_center_bottom);
	signature1.push_back(right_center_bottom);
	// Push the new signature vector into the signatures vector
	return signature1;
}

vector<Point2f> extractFeaturesTest(Mat signature) {
	// Apply Gaussian filter to remove noise
	Mat signature_filtered;
	GaussianBlur(signature, signature_filtered, Size(5, 5), 0, 0);

	// Apply thresholding to convert to binary image
	Mat signature_binary;
	threshold(signature_filtered, signature_binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	normalize(signature_binary, signature_binary, 0, 1, NORM_MINMAX, CV_32F);

	// Apply Canny edge detection to extract boundary
	Mat signature_edges = signature.clone();

	// Find the center of the signature image
	Moments m = moments(signature_edges, true);
	Point center(m.m10 / m.m00, m.m01 / m.m00);

	// Remove lines longer than threshold
	double threshold = 150;
	// Find lines in the image
	vector<Vec4i> lines;
	HoughLinesP(signature_edges, lines, 1, CV_PI / 180, threshold, 0, 0);

	// Draw lines over the image
	Mat signature_cleaned = signature_edges.clone();
	vector<vector<Point>> polylines;
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		Point pt1(line[0], line[1]);
		Point pt2(line[2], line[3]);
		polylines.push_back({ pt1, pt2 });
	}

	cv::polylines(signature_cleaned, polylines, true, Scalar(0), 3, LINE_AA);

	circle(signature_cleaned, center, 5, Scalar(255, 0, 255), -1);

	int midx = center.x;
	Mat left_half = signature_cleaned(Rect(0, 0, midx, signature_cleaned.rows));
	// Find center of the left half
	Moments left_m = moments(left_half, true);
	Point left_center(left_m.m10 / left_m.m00, left_m.m01 / left_m.m00);
	circle(signature_cleaned, left_center, 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned, Point(0, left_center.y), Point(center.x, left_center.y), Scalar(255, 255, 255), 2);

	// Find center of the right half
	Mat right_half = signature_cleaned(Rect(midx, 0, signature_cleaned.cols - midx, signature_cleaned.rows));
	Moments right_m = moments(right_half, true);
	Point right_center(midx + right_m.m10 / right_m.m00, right_m.m01 / right_m.m00);
	circle(signature_cleaned, right_center, 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned, Point(signature_cleaned.cols, right_center.y), Point(center.x, right_center.y), Scalar(255, 255, 255), 2);


	line(signature_cleaned, Point(center.x, 0), Point(center.x, signature_filtered.rows), Scalar(255, 255, 255), 2);

	Mat top_left = signature_cleaned(Rect(0, 0, midx, left_center.y));
	Moments top_left_m = moments(top_left, true);
	Point top_left_center(top_left_m.m10 / top_left_m.m00, top_left_m.m01 / top_left_m.m00);
	circle(signature_cleaned, top_left_center, 5, Scalar(255, 0, 255), -1);

	Mat top_right = signature_cleaned(Rect(midx, 0, signature_cleaned.cols - midx, top_left.rows));
	Moments top_right_m = moments(top_right, true);
	Point top_right_center(midx + top_right_m.m10 / top_right_m.m00, top_right_m.m01 / top_right_m.m00);
	circle(signature_cleaned, top_right_center, 5, Scalar(255, 0, 255), -1);

	// Extract bottom left quarter
	Mat bottom_left = signature_cleaned(Rect(0, left_center.y, midx, signature_cleaned.rows - left_center.y));
	Moments bottom_left_m = moments(bottom_left, true);
	Point bottom_left_center(bottom_left_m.m10 / bottom_left_m.m00, left_center.y + bottom_left_m.m01 / bottom_left_m.m00);
	circle(signature_cleaned, bottom_left_center, 5, Scalar(255, 0, 255), -1);

	// Extract bottom right quarter
	Mat bottom_right = signature_cleaned(Rect(midx, left_center.y, signature_cleaned.cols - midx, signature_cleaned.rows - left_center.y));
	Moments bottom_right_m = moments(bottom_right, true);
	Point bottom_right_center(midx + bottom_right_m.m10 / bottom_right_m.m00, left_center.y + bottom_right_m.m01 / bottom_right_m.m00);
	circle(signature_cleaned, bottom_right_center, 5, Scalar(255, 0, 255), -1);

	Mat signature_cleaned_2 = signature_cleaned.clone();
	Point center_1(m.m10 / m.m00, m.m01 / m.m00);
	circle(signature_cleaned_2, center_1, 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned_2, Point(0, center_1.y), Point(signature_filtered.cols, center_1.y), Scalar(255, 255, 255), 2);
	midx = center_1.x;

	Mat top_half = signature_cleaned_2(Rect(0, 0, signature_cleaned_2.cols, center.y));
	Moments top_m = moments(top_half, true);
	Point top_center(top_m.m10 / top_m.m00, top_m.m01 / top_m.m00);
	circle(top_half, top_center, 5, Scalar(255, 0, 255), -1);
	line(top_half, Point(top_center.x, 0), Point(top_center.x, center.y), Scalar(255, 255, 255), 2);

	int midy = center.y;
	Mat bottom_half = signature_cleaned_2(Rect(0, midy, signature_cleaned_2.cols, signature_cleaned_2.rows - midy));
	Moments bottom_m = moments(bottom_half, true);
	Point bottom_center(bottom_m.m10 / bottom_m.m00, bottom_m.m01 / bottom_m.m00);
	circle(signature_cleaned_2, bottom_center + Point(0, midy), 5, Scalar(255, 0, 255), -1);
	line(signature_cleaned_2, Point(bottom_center.x, midy), Point(bottom_center.x, signature_filtered.rows), Scalar(255, 255, 255), 2);

	Rect left_roi_top(0, 0, top_half.cols / 2, top_half.rows);
	Mat left_half_top = top_half(left_roi_top);

	// Compute moments and mass center for left section
	Moments left_moments_top = moments(left_half_top, true);
	Point left_center_top(left_moments_top.m10 / left_moments_top.m00, left_moments_top.m01 / left_moments_top.m00);
	circle(top_half, left_center_top, 5, Scalar(255, 255, 255), -1);

	Rect right_roi_top(top_half.cols / 2, 0, top_half.cols / 2, top_half.rows);
	Mat right_half_top = top_half(right_roi_top);

	// Compute moments and mass center for right section
	Moments right_moments_top = moments(right_half_top, true);
	Point right_center_top(right_moments_top.m10 / right_moments_top.m00 + top_half.cols / 2, right_moments_top.m01 / right_moments_top.m00);
	circle(top_half, right_center_top, 5, Scalar(255, 0, 255), -1);



	Rect left_roi_bottom(0, 0, bottom_half.cols / 2, bottom_half.rows);
	Mat left_half_bottom = bottom_half(left_roi_bottom);
	Moments left_moments_bottom = moments(left_half_bottom, true);
	Point left_center_bottom(left_moments_bottom.m10 / left_moments_bottom.m00, left_moments_bottom.m01 / left_moments_bottom.m00);
	circle(bottom_half, left_center_bottom, 5, Scalar(255, 255, 255), -1);

	Rect right_roi_bottom(bottom_half.cols / 2, 0, bottom_half.cols / 2, top_half.rows);

	Mat right_half_bottom = bottom_half(right_roi_bottom);
	Moments right_moments_bottom = moments(right_half_bottom, true);
	Point right_center_bottom(right_moments_bottom.m10 / right_moments_bottom.m00 + bottom_half.cols / 2, right_moments_bottom.m01 / right_moments_bottom.m00);
	circle(bottom_half, right_center_bottom, 5, Scalar(255, 0, 255), -1);


	std::vector<cv::Point2f> signature1;
	signature1.push_back(left_center);
	signature1.push_back(right_center);
	signature1.push_back(top_left_center);
	signature1.push_back(top_right_center);
	signature1.push_back(bottom_left_center);
	signature1.push_back(bottom_right_center);
	signature1.push_back(top_center);
	signature1.push_back(bottom_center);
	signature1.push_back(left_center_top);
	signature1.push_back(right_center_top);
	signature1.push_back(left_center_bottom);
	signature1.push_back(right_center_bottom);
	// Push the new signature vector into the signatures vector
	return signature1;
}


float euclidean_distance(const vector<Point2f>& v1, const vector<Point2f>& v2) {
	float sum = 0.0;
	for (int i = 0; i < v1.size(); i++) {
		float dx = v1[i].x - v2[i].x;
		float dy = v1[i].y - v2[i].y;
		sum += dx * dx + dy * dy;
	}
	return sqrt(sum);
}


float cosine_distance(const std::vector<Point2f>& v1, const std::vector<Point2f>& v2) {
	float dotProduct = 0.0;
	float magnitudeV1 = 0.0;
	float magnitudeV2 = 0.0;

	for (int i = 0; i < v1.size(); i++) {
		dotProduct += v1[i].x * v2[i].x + v1[i].y * v2[i].y;
		magnitudeV1 += v1[i].x * v1[i].x + v1[i].y * v1[i].y;
		magnitudeV2 += v2[i].x * v2[i].x + v2[i].y * v2[i].y;
	}

	if (magnitudeV1 == 0.0 || magnitudeV2 == 0.0) {
		return 0.0;  // Handle division by zero case
	}

	return 1 - (dotProduct / (std::sqrt(magnitudeV1) * std::sqrt(magnitudeV2)));
}
// Function to classify a test signature using the KNN algorithm

bool knn_classify(const std::vector<std::pair<std::vector<Point2f>, std::string>>& train_set,
	const std::vector<Point2f>& test_signature, int k, const std::string& distance_metric) {

	// Compute the distances between the test signature and each signature in the training set
	std::vector<std::pair<float, std::string>> distances;
	for (const auto& train_signature : train_set) {
		float distance;
		if (distance_metric == "euclidean") {
			distance = euclidean_distance(train_signature.first, test_signature);
		}
		else if (distance_metric == "cosine") {
			distance = cosine_distance(train_signature.first, test_signature);
		}
		else {
			throw std::runtime_error("Invalid distance metric");
		}
		distances.push_back(std::make_pair(distance, train_signature.second));
	}

	// Sort the distances in ascending order
	std::sort(distances.begin(), distances.end());

	// Take the labels of the k nearest neighbors
	int genuine_count = 0, forged_count = 0;
	for (int i = 0; i < k; i++) {
		if (distances[i].second == "genuine") {
			genuine_count++;
		}
		else {
			forged_count++;
		}
	}

	// Classify the test signature based on the majority vote
	return genuine_count > forged_count;
}
float calculateAccuracy(const vector<pair<vector<Point2f>, string>>train_set, int k, const string& distance_metric) {
	int genuine_count = 0, forged_count = 0;
	int total_samples = 0;

	for (const auto& signature : train_set) {
		// Extract signature data
		const vector<Point2f>& featureVector = signature.first;
		const string& label = signature.second;

		// Classify the signature using the KNN classifier
		bool isGenuine = knn_classify(train_set, featureVector, k, distance_metric);

		// Compare the predicted label with the ground truth label
		if ((isGenuine && label == "genuine") || (!isGenuine && label == "forged")) {
			if (label == "genuine") {
				genuine_count++;
			}
			else {
				forged_count++;
			}
		}
		total_samples++;
	}

	// Calculate accuracy
	float accuracy = static_cast<float>(genuine_count) / total_samples * 100;

	return accuracy;
}
void saveDataToFile(const std::string& filename, const std::vector<std::pair<int, float>>& dataPoints) {
	std::ofstream dataFile(filename);
	if (!dataFile.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	// Write the data points to the file
	for (const auto& point : dataPoints) {
		dataFile << point.first << " " << point.second << "\n";
	}

	dataFile.close();
}

void createGraph(const std::string& filename, const std::string& distance_metric, const int user, vector<pair<vector<Point2f>, string>> train_set) {

	vector<pair<vector<Point2f>, string>> test_set;
	std::vector<std::pair<int, float>> dataPoints;
	std::vector<int> k_values = { 11,13,15 };
	for (int t = 0; t < k_values.size(); t++) {
		for (int i = 21; i < 46; i++) {
			createTestSignature(i, user);
			Mat testSignatureImg = imread("signature.jpg", IMREAD_GRAYSCALE);
			vector<Point2f> testFeatureVector = extractFeaturesTest(testSignatureImg);

			// Classify the test signature using the KNN classifier
			bool isGenuine = knn_classify(train_set, testFeatureVector, k_values[t], distance_metric);
			string label = (isGenuine) ? "genuine" : "forged";
			test_set.push_back(make_pair(testFeatureVector, label));
			// Print the classification result
			if (isGenuine) {
				cout << "The test signature is genuine." << endl;
			}
			else {
				cout << "The test signature is forged." << endl;
			}
		}
		float accuracy = calculateAccuracy(test_set, k_values[t], distance_metric);
		dataPoints.push_back(make_pair(k_values[t], accuracy));
		test_set.clear();

		cout << "Accuracy: " << accuracy << "%" << endl;
		std::string s = "plot '-' with lines title '" + distance_metric + "'\n";
	}
	saveDataToFile(filename, dataPoints);
}
int main()
{
	printf("Choose user: ");
	int user;
	scanf("%d", &user);

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
				vector<pair<vector<Point2f>, string>> train_set;
				createSignatureGenuine(user);
				createSignatureForged(user);

				// Adding the first set of genuine signatures with label "genuine"
				for (int i = 46; i <=65; i++) {
					string ss= "signature_" + to_string(i)+ ".jpg";
					Mat signatureImg = imread(ss, IMREAD_GRAYSCALE);
					vector<Point2f> featureVector = extractFeatures(signatureImg);
					train_set.push_back(make_pair(featureVector, "genuine"));
				}

				// Adding the first set of forged signatures with label "forged"
				for (int i = 1; i <=20; i++) {
					string ss = "signature_" + to_string(i) + ".jpg";
					Mat signatureImg = imread(ss, IMREAD_GRAYSCALE);
					vector<Point2f> featureVector = extractFeatures(signatureImg);
					train_set.push_back(make_pair(featureVector, "forged"));
				}

				// Train the KNN classifier
				// Load the test signature image and extract its feature vector

				createGraph("cosinus.txt", "cosine",user,train_set);
				createGraph("euclidian.txt", "euclidean",user,train_set);
				std::cin.ignore(); // Ignore any characters the user may have already typed
				std::cin.get();
				break;

		}
	}while (op!=0);

	return 0;
}



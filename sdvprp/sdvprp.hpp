#pragma once
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <random>
#include <cmath>
#include <algorithm>

struct EdgeSegment {
    std::vector<cv::Point> points;
};

struct GradientResult {
    cv::Mat maxGrad; 
    cv::Mat gx;     
    cv::Mat gy;       
};



GradientResult computeColorGradient(const cv::Mat& colorImage);
cv::Mat thresholdEdges(const cv::Mat& gradientMagnitude, float lowThresh, float highThresh);
std::vector<EdgeSegment> followEdges(const cv::Mat& edgeMap, const cv::Mat& gradX, const cv::Mat& gradY, float maxAngleDiffDeg, float collinearityThreshold = 20.0f);
std::vector<EdgeSegment> filterHorizontalSegments(const std::vector<EdgeSegment>& segments, float minLength = 50.0f, float maxAngleFromHorizontalDeg = 20.0f);

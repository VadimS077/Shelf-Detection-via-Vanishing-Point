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

struct CandidateVP {
    cv::Point2f location;
    float score = 0.0f;
    std::vector<const EdgeSegment*> associatedSegments;
};
struct ShelfCandidate {
    float y_mid;
    float weight;
    std::vector<const EdgeSegment*> supportingSegments;
    cv::Point2f leftPoint;
    cv::Point2f rightPoint;
};

GradientResult computeColorGradient(const cv::Mat& colorImage);
cv::Mat thresholdEdges(const cv::Mat& gradientMagnitude, float lowThresh=0.2f);
std::vector<EdgeSegment> followEdges(const cv::Mat& edgeMap, const cv::Mat& gradX, const cv::Mat& gradY, float maxAngleDiffDeg=15.0f, float collinearityThreshold = 20.0f);
std::vector<EdgeSegment> filterHorizontalSegments(const std::vector<EdgeSegment>& segments, float minLength = 70.0f, float maxAngleFromHorizontalDeg = 40.0f);
cv::Point2f computeVanishingPoint(const cv::Point2f& a1, const cv::Point2f& a2,
    const cv::Point2f& b1, const cv::Point2f& b2);
CandidateVP findBestVanishingPoint(const std::vector<EdgeSegment>& horizSegs, float angleThresholdDeg = 3.0f, int maxPairs = 1000);
std::vector<ShelfCandidate> findShelves(const std::vector<EdgeSegment>& horizSegs,
    const cv::Point2f& vp,
    const cv::Size& imageSize,
    float scaleX, float scaleY,
    float binSize = 2.0f,
    float minWeight = 300.0f);
std::vector<ShelfCandidate> mergeCloseShelves(std::vector<ShelfCandidate>& candidates,
    float mergeThreshold = 15.0f);

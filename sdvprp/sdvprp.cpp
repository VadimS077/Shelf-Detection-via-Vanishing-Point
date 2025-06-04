#include "sdvprp.hpp"

GradientResult computeColorGradient(const cv::Mat& colorImage) {
    // BGR
    std::vector<cv::Mat> channels;
    cv::split(colorImage, channels);

    cv::Mat gradB_x, gradB_y, gradG_x, gradG_y, gradR_x, gradR_y;
    cv::Sobel(channels[0], gradB_x, CV_32F, 1, 0); // ∂B/∂x
    cv::Sobel(channels[0], gradB_y, CV_32F, 0, 1); // ∂B/∂y
    cv::Sobel(channels[1], gradG_x, CV_32F, 1, 0); // ∂G/∂x
    cv::Sobel(channels[1], gradG_y, CV_32F, 0, 1); // ∂G/∂y
    cv::Sobel(channels[2], gradR_x, CV_32F, 1, 0); // ∂R/∂x
    cv::Sobel(channels[2], gradR_y, CV_32F, 0, 1); // ∂R/∂y

    cv::Mat magB, magG, magR;
    cv::magnitude(gradB_x, gradB_y, magB);
    cv::magnitude(gradG_x, gradG_y, magG);
    cv::magnitude(gradR_x, gradR_y, magR);

    cv::Mat maxGrad;
    cv::max(magB, magG, maxGrad);
    cv::max(maxGrad, magR, maxGrad);

//Создание gx и gy, соответствующих каналу с максимальным градиентом
    cv::Mat gx = cv::Mat::zeros(maxGrad.size(), CV_32F);
    cv::Mat gy = cv::Mat::zeros(maxGrad.size(), CV_32F);

    for (int y = 0; y < maxGrad.rows; ++y) {
        for (int x = 0; x < maxGrad.cols; ++x) {
            float b = magB.at<float>(y, x);
            float g = magG.at<float>(y, x);
            float r = magR.at<float>(y, x);
            if (b >= g && b >= r) {
                gx.at<float>(y, x) = gradB_x.at<float>(y, x);
                gy.at<float>(y, x) = gradB_y.at<float>(y, x);
            }
            else if (g >= b && g >= r) {
                gx.at<float>(y, x) = gradG_x.at<float>(y, x);
                gy.at<float>(y, x) = gradG_y.at<float>(y, x);
            }
            else {
                gx.at<float>(y, x) = gradR_x.at<float>(y, x);
                gy.at<float>(y, x) = gradR_y.at<float>(y, x);
            }
        }
    }

    return { maxGrad, gx, gy };
}

cv::Mat thresholdEdges(const cv::Mat& gradientMagnitude, float lowThresh, float highThresh) {
    cv::Mat edges;
    cv::Mat strongEdges = gradientMagnitude > highThresh;
    cv::Mat weakEdges = (gradientMagnitude > lowThresh) & (gradientMagnitude <= highThresh);

    edges = cv::Mat::zeros(gradientMagnitude.size(), CV_8U);
    edges.setTo(100, weakEdges);
    edges.setTo(255, strongEdges);
    return edges;
}
std::vector<EdgeSegment> followEdges(const cv::Mat& edgeMap, const cv::Mat& gradX, const cv::Mat& gradY, float maxAngleDiffDeg, float collinearityThreshold) {
    cv::Mat visited = cv::Mat::zeros(edgeMap.size(), CV_8U);
    std::vector<EdgeSegment> segments;

    int rows = edgeMap.rows, cols = edgeMap.cols;
    const float maxAngleDiff = maxAngleDiffDeg * CV_PI / 180.0f;

    auto computeAngle = [&](cv::Point pt) {
        float gx = gradX.at<float>(pt), gy = gradY.at<float>(pt);
        return std::atan2(gy, gx);
        };

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            if (edgeMap.at<uchar>(y, x) >= 100 && !visited.at<uchar>(y, x)) {
                std::deque<cv::Point> segmentPoints;
                float refAngle = computeAngle(cv::Point(x, y));
                segmentPoints.push_back({ x, y });
                visited.at<uchar>(y, x) = 1;

                std::deque<cv::Point> queue = { {x, y} };
                while (!queue.empty()) {
                    cv::Point p = queue.front();
                    queue.pop_front();

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            int nx = p.x + dx, ny = p.y + dy;
                            if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
                            if (visited.at<uchar>(ny, nx)) continue;
                            if (edgeMap.at<uchar>(ny, nx) < 100) continue;

                            float angle = computeAngle(cv::Point(nx, ny));
                            if (std::abs(angle - refAngle) < maxAngleDiff) {
                                visited.at<uchar>(ny, nx) = 1;
                                queue.push_back({ nx, ny });
                                segmentPoints.push_back({ nx, ny });
                            }
                        }
                    }
                }

                // Проверка коллинеарности
                if (segmentPoints.size() > 5) {
                    cv::Mat pointsMat(segmentPoints.size(), 2, CV_32F);
                    for (size_t i = 0; i < segmentPoints.size(); ++i) {
                        pointsMat.at<float>(i, 0) = segmentPoints[i].x;
                        pointsMat.at<float>(i, 1) = segmentPoints[i].y;
                    }

                    // Вычисление PCA
                    cv::PCA pca(pointsMat, cv::Mat(), cv::PCA::DATA_AS_ROW);
                    float lambda1 = pca.eigenvalues.at<float>(0);
                    float lambda2 = pca.eigenvalues.at<float>(1);

                    if (lambda1 / lambda2 > collinearityThreshold) {
                        segments.push_back({ std::vector<cv::Point>(segmentPoints.begin(), segmentPoints.end()) });
                    }
                }
            }
        }
    }

    return segments;
}
std::vector<EdgeSegment> filterHorizontalSegments(const std::vector<EdgeSegment>& segments, float minLength, float maxAngleFromHorizontalDeg) {
    std::vector<EdgeSegment> filtered;
    float maxAngle = maxAngleFromHorizontalDeg * CV_PI / 180.0f;

    for (const auto& seg : segments) {
        if (seg.points.size() < 2) continue;

        cv::Point2f p1 = seg.points.front();
        cv::Point2f p2 = seg.points.back();

        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float length = std::sqrt(dx * dx + dy * dy);
        float angle = std::atan2(std::abs(dy), std::abs(dx));  // угол между отрезком и горизонталью

        if (length >= minLength && angle < maxAngle) {
            filtered.push_back(seg);
        }
    }

    return filtered;
}

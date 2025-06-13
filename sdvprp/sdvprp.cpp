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


cv::Point2f computeVanishingPoint(const cv::Point2f& a1, const cv::Point2f& a2,
    const cv::Point2f& b1, const cv::Point2f& b2)
{
    //figure2
    cv::Vec3f l1 = cv::Vec3f(a1.y - a2.y, a2.x - a1.x, a1.x * a2.y - a2.x * a1.y);
    cv::Vec3f l2 = cv::Vec3f(b1.y - b2.y, b2.x - b1.x, b1.x * b2.y - b2.x * b1.y);

    cv::Vec3f vp = l1.cross(l2);

    if (std::abs(vp[2]) < 1e-5) return cv::Point2f(-1, -1); // если в бесконечность ушла

    return cv::Point2f(vp[0] / vp[2], vp[1] / vp[2]);
}


CandidateVP findBestVanishingPoint(const std::vector<EdgeSegment>& horizSegs, float angleThresholdDeg, int maxPairs) {
    std::vector<CandidateVP> candidates;
    float angleThreshold = angleThresholdDeg * CV_PI / 180.0f;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, (int)horizSegs.size() - 1);

    for (int i = 0; i < maxPairs; ++i) {
        int idx1 = dist(rng), idx2 = dist(rng);
        if (idx1 == idx2) continue;

        auto& seg1 = horizSegs[idx1];
        auto& seg2 = horizSegs[idx2];
        if (seg1.points.size() < 2 || seg2.points.size() < 2) continue;

        cv::Point2f vp = computeVanishingPoint(seg1.points.front(), seg1.points.back(),
            seg2.points.front(), seg2.points.back());

        CandidateVP cand;
        cand.location = vp;

        for (auto& seg : horizSegs) {
            cv::Point2f mid = 0.5f * (seg.points.front() + seg.points.back());
            cv::Point2f dir1 = seg.points.back() - seg.points.front();
            cv::Point2f dir2 = mid - vp;


            float cos_angle = dir1.dot(dir2) / (cv::norm(dir1) * cv::norm(dir2) + 1e-5f);
            if (std::abs(cos_angle) > std::cos(angleThreshold)) {
                cand.score += (float)cv::norm(dir1);
                cand.associatedSegments.push_back(&seg);
            }
        }

        candidates.push_back(cand);
    }

    return *std::max_element(candidates.begin(), candidates.end(),
        [](const CandidateVP& a, const CandidateVP& b) {
            return a.score < b.score;
        });
}




std::vector<ShelfCandidate> findShelves(const std::vector<EdgeSegment>& horizSegs,
    const cv::Point2f& vp,
    const cv::Size& imageSize,
    float scaleX, float scaleY,
    float binSize,
    float minWeight)
{
    const int midX = imageSize.width / 2;
    const int numBins = static_cast<int>(imageSize.height / binSize) + 1;
    std::vector<float> votes(numBins, 0.0f);

    for (const auto& seg : horizSegs) {
        cv::Point2f mid_small = 0.5f * (seg.points.front() + seg.points.back());
        cv::Point2f mid_full(mid_small.x * scaleX, mid_small.y * scaleY);


        cv::Point2f dir = mid_full - vp;
        if (std::abs(dir.x) < 1e-5f) continue;

        float t = (midX - vp.x) / dir.x;
        if (t <= 0) continue;

        float y_intersect = vp.y + t * dir.y;
        if (y_intersect < 0 || y_intersect >= imageSize.height) continue;

        cv::Point2f p1_full(seg.points.front().x * scaleX, seg.points.front().y * scaleY);
        cv::Point2f p2_full(seg.points.back().x * scaleX, seg.points.back().y * scaleY);
        float seg_length = cv::norm(p1_full - p2_full);

        int bin_idx = static_cast<int>(y_intersect / binSize);
        if (bin_idx >= 0 && bin_idx < numBins) {
            votes[bin_idx] += seg_length;
        }
    }


    std::vector<ShelfCandidate> candidates;
    for (int i = 1; i < numBins - 1; ++i) {
        if (votes[i] > votes[i - 1] &&
            votes[i] > votes[i + 1] &&
            votes[i] >= minWeight)
        {
            ShelfCandidate cand;
            cand.y_mid = (i + 0.5f) * binSize;
            cand.weight = votes[i];
            candidates.push_back(cand);
        }
    }


    for (auto& cand : candidates) {
        float total_weight = 0.0f;
        float sum_y = 0.0f;
        cand.supportingSegments.clear();


        for (const auto& seg : horizSegs) {
            cv::Point2f mid_small = 0.5f * (seg.points.front() + seg.points.back());
            cv::Point2f mid_full(mid_small.x * scaleX, mid_small.y * scaleY);
            cv::Point2f dir = mid_full - vp;

            if (std::abs(dir.x) < 1e-5f) continue;
            float t = (midX - vp.x) / dir.x;
            if (t <= 0) continue;

            float y_intersect = vp.y + t * dir.y;
            if (y_intersect < 0 || y_intersect >= imageSize.height) continue;

            if (std::abs(y_intersect - cand.y_mid) <= binSize * 1.5f) {
                cv::Point2f p1_full(seg.points.front().x * scaleX, seg.points.front().y * scaleY);
                cv::Point2f p2_full(seg.points.back().x * scaleX, seg.points.back().y * scaleY);
                float seg_length = cv::norm(p1_full - p2_full);

                sum_y += y_intersect * seg_length;
                total_weight += seg_length;
                cand.supportingSegments.push_back(&seg);
            }
        }

        if (total_weight > 0) {
            cand.y_mid = sum_y / total_weight;
            cand.weight = total_weight;

            cv::Point2f dir_to_mid(midX - vp.x, cand.y_mid - vp.y);
            float t_left = (0 - vp.x) / dir_to_mid.x;
            float y_left = vp.y + t_left * dir_to_mid.y;
            float t_right = (imageSize.width - 1 - vp.x) / dir_to_mid.x;
            float y_right = vp.y + t_right * dir_to_mid.y;

            cand.leftPoint = cv::Point2f(0, y_left);
            cand.rightPoint = cv::Point2f(imageSize.width - 1, y_right);
        }
    }

    auto it = std::remove_if(candidates.begin(), candidates.end(),
        [](const ShelfCandidate& c) { return c.weight == 0; });
    candidates.erase(it, candidates.end());

    std::sort(candidates.begin(), candidates.end(),
        [](const ShelfCandidate& a, const ShelfCandidate& b) {
            return a.y_mid < b.y_mid;
        });

    return candidates;
}


std::vector<ShelfCandidate> mergeCloseShelves(std::vector<ShelfCandidate>& candidates,
    float mergeThreshold) {
    if (candidates.empty()) return {};


    std::vector<ShelfCandidate> merged;
    ShelfCandidate current = candidates[0];

    for (size_t i = 1; i < candidates.size(); ++i) {
        if (std::abs(candidates[i].y_mid - current.y_mid) < mergeThreshold) {
            float totalWeight = current.weight + candidates[i].weight;
            current.y_mid = (current.y_mid * current.weight +
                candidates[i].y_mid * candidates[i].weight) / totalWeight;

            current.supportingSegments.insert(current.supportingSegments.end(),
                candidates[i].supportingSegments.begin(),
                candidates[i].supportingSegments.end());

            current.leftPoint = (current.leftPoint * current.weight +
                candidates[i].leftPoint * candidates[i].weight) / totalWeight;
            current.rightPoint = (current.rightPoint * current.weight +
                candidates[i].rightPoint * candidates[i].weight) / totalWeight;

            current.weight = totalWeight;
        }
        else {
            merged.push_back(current);
            current = candidates[i];
        }
    }
    merged.push_back(current);

    return merged;
}

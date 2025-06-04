#include <sdvprp.hpp> 



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
struct CandidateVP {
    cv::Point2f location;
    float score = 0.0f;
    std::vector<const EdgeSegment*> associatedSegments;
};

CandidateVP findBestVanishingPoint(const std::vector<EdgeSegment>& horizSegs, float angleThresholdDeg = 3.0f, int maxPairs = 500) {
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

            float angle = std::acos(
                (dir1.dot(dir2)) / (cv::norm(dir1) * cv::norm(dir2) + 1e-5f)
            );
            if (std::abs(angle) < angleThreshold) {
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



int main() {
    cv::Mat image = cv::imread("C:\\Users\\vadim\\source\\repos\\CourseWork\\100.jpg");

    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0.8);

    auto gradResult = computeColorGradient(small);
    cv::Mat normalizedGrad;
    cv::normalize(gradResult.maxGrad, normalizedGrad, 0, 1, cv::NORM_MINMAX, CV_32F);
    cv::Mat edgeMap = thresholdEdges(normalizedGrad, 0.2, 0.8);

    auto segments = followEdges(edgeMap, gradResult.gx, gradResult.gy,20);
    auto horizontalSegments = filterHorizontalSegments(segments, 40.0f, 45.0f);

//Визуализация отрезков
    cv::Mat vis = image.clone();
    float scaleX = static_cast<float>(image.cols) / small.cols;
    float scaleY = static_cast<float>(image.rows) / small.rows;

    for (const auto& seg : horizontalSegments) {
        for (size_t i = 1; i < seg.points.size(); ++i) {
            cv::Point p1(seg.points[i - 1].x * scaleX, seg.points[i - 1].y * scaleY);
            cv::Point p2(seg.points[i].x * scaleX, seg.points[i].y * scaleY);
            cv::line(vis, p1, p2, cv::Scalar(0, 255, 0), 2);
        }
    }

    auto vpResult = findBestVanishingPoint(horizontalSegments, 3.0f, 500);

    cv::Point2f vp = vpResult.location;
    cv::Point2f vpOnFullSize(vp.x * scaleX, vp.y * scaleY);

    if (vpOnFullSize.inside(cv::Rect(0, 0, vis.cols, vis.rows))) {
        cv::circle(vis, vpOnFullSize, 6, cv::Scalar(0, 0, 255), -1);
    }
    else {

        cv::Point2f imageCenter(vis.cols / 2.0f, vis.rows / 2.0f);

        cv::Point2f dir = vpOnFullSize - imageCenter;

        float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len > 1e-3f) {
            //стрелка в направлении vp
            cv::Point2f dirNorm = dir * (1000.0f / len);
            cv::Point2f p1 = imageCenter - dirNorm; 
            cv::Point2f p2 = imageCenter + dirNorm;

            cv::arrowedLine(vis, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0.3);
        }

        std::cout << "Vanishing point out of bounds: " << vpOnFullSize << std::endl;
    }
    std::cout << "Vanishing point: " << vpOnFullSize << std::endl;


    cv::imwrite("C:\\Users\\vadim\\source\\repos\\CourseWork\\st1.jpg", vis);

    return 0;
}

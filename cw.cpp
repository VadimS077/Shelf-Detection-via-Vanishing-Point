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


            float cos_angle = dir1.dot(dir2) / (cv::norm(dir1) * cv::norm(dir2) + 1e-5f);
            if (std::abs(cos_angle) > std::cos(angleThreshold)){
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


struct ShelfCandidate {
    float y_mid; 
    float weight; 
    std::vector<const EdgeSegment*> supportingSegments;
    cv::Point2f leftPoint; 
    cv::Point2f rightPoint; 
};

std::vector<ShelfCandidate> findShelves(const std::vector<EdgeSegment>& horizSegs,
    const cv::Point2f& vp,
    const cv::Size& imageSize,
    float scaleX, float scaleY,
    float binSize = 5.0f,
    float minWeight = 50.0f)
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
    float mergeThreshold = 15.0f) {
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


int main() {
    cv::Mat image = cv::imread("C:\\Users\\vadim\\source\\repos\\CourseWork\\007.jpg");

    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0.8);

    auto gradResult = computeColorGradient(small);
    cv::Mat normalizedGrad;
    cv::normalize(gradResult.maxGrad, normalizedGrad, 0, 1, cv::NORM_MINMAX, CV_32F);
    cv::Mat edgeMap = thresholdEdges(normalizedGrad, 0.2, 0.8);

    auto segments = followEdges(edgeMap, gradResult.gx, gradResult.gy,20);
    auto horizontalSegments = filterHorizontalSegments(segments, 70.0f, 45.0f);

//Визуализация отрезков
    cv::Mat vis = image.clone();
    cv::Mat visa = image.clone();
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
            cv::Point2f dirNorm = dir * (10000.0f / len);
            cv::Point2f p1 = imageCenter - dirNorm; 
            cv::Point2f p2 = imageCenter + dirNorm;

            cv::arrowedLine(vis, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0.3);
        }

        std::cout << "Vanishing point out of bounds: " << vpOnFullSize << std::endl;
    }
    std::vector<ShelfCandidate> shelves = findShelves(
        horizontalSegments,
        vpOnFullSize,
        image.size(),
        scaleX, scaleY,
        2.0f,   
        300.0f   
    );

    float mergeThreshold = 0.02f * image.rows;
    std::vector<ShelfCandidate> finalShelves = mergeCloseShelves(shelves, mergeThreshold);


    // Визуализация полок
    for (const auto& shelf : finalShelves) {
        cv::line(visa, shelf.leftPoint, shelf.rightPoint, cv::Scalar(0, 0, 255), 3);

    }


    cv::imwrite("C:\\Users\\vadim\\source\\repos\\CourseWork\\st4.jpg", visa);
    cv::imwrite("C:\\Users\\vadim\\source\\repos\\CourseWork\\st2.jpg", vis);


    return 0;
}

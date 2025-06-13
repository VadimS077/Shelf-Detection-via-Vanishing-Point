#include <sdvprp.hpp> 




int main() {
    cv::Mat image = cv::imread("C:\\Users\\vadim\\source\\repos\\CourseWork\\train_8.jpg");

    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0.8);

    auto gradResult = computeColorGradient(small);
    cv::Mat normalizedGrad;
    cv::normalize(gradResult.maxGrad, normalizedGrad, 0, 1, cv::NORM_MINMAX, CV_32F);
    cv::Mat edgeMap = thresholdEdges(normalizedGrad, 0.2, 0.8);

    auto segments = followEdges(edgeMap, gradResult.gx, gradResult.gy,20);
    auto horizontalSegments = filterHorizontalSegments(segments);

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

    auto vpResult = findBestVanishingPoint(horizontalSegments);
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
        scaleX, scaleY 
    );

    float mergeThreshold = 0.015f * image.rows;
    std::vector<ShelfCandidate> finalShelves = mergeCloseShelves(shelves, mergeThreshold);


    // Визуализация полок
    for (const auto& shelf : finalShelves) {
        cv::line(visa, shelf.leftPoint, shelf.rightPoint, cv::Scalar(0, 0, 255), 3);

    }


    cv::imwrite("C:\\Users\\vadim\\source\\repos\\CourseWork\\st1.jpg", visa);
    cv::imwrite("C:\\Users\\vadim\\source\\repos\\CourseWork\\st2.jpg", vis);


    return 0;
}

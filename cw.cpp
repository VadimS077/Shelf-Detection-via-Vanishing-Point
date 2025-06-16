#include <sdvprp.hpp> 




int main(int argc, char* argv[]) {
   /* if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
            << " path_to_image path_to_result "
            << "[lowThresh] [highThresh] [minWeight] [minLength] [mergeThreshold] [binSize]\n";
        return 1;
    }*/

    std::string imagePath = "C:\\Users\\vadim\\source\\repos\\Coursework\\101.jpg";//argv[1];
    std::string outputPath = "C:\\Users\\vadim\\source\\repos\\Coursework\\res3.jpg";//argv[2];

    float lowThresh = 0.2f;
    float highThresh = 0.8f;
    float minWeight = 300.0f;
    float minLength = 70.0f;
    float mergeThreshold = 0.015f;
    float binSize = 2.0f;

    if (argc > 3) lowThresh = std::stof(argv[3]);
    if (argc > 4) highThresh = std::stof(argv[4]);
    if (argc > 5) minWeight = std::stof(argv[5]);
    if (argc > 6) minLength = std::stof(argv[6]);
    if (argc > 7) mergeThreshold = std::stof(argv[7]);
    if (argc > 8) binSize = std::stof(argv[8]);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << imagePath << "\n";
        return 1;
    }

    cv::Mat small;
    cv::resize(image, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::GaussianBlur(small, small, cv::Size(5, 5), 0.8);

    auto gradResult = computeColorGradient(small);
    cv::Mat normalizedGrad;
    cv::normalize(gradResult.maxGrad, normalizedGrad, 0, 1, cv::NORM_MINMAX, CV_32F);
    cv::Mat edgeMap = thresholdEdges(normalizedGrad, lowThresh, highThresh);

    auto segments = followEdges(edgeMap, gradResult.gx, gradResult.gy,15);
    auto horizontalSegments = filterHorizontalSegments(segments,minLength);

    cv::Mat visa = image.clone();
    float scaleX = static_cast<float>(image.cols) / small.cols;
    float scaleY = static_cast<float>(image.rows) / small.rows;

    auto vpResult = findBestVanishingPoint(horizontalSegments);
    cv::Point2f vp = vpResult.location;
    cv::Point2f vpOnFullSize(vp.x * scaleX, vp.y * scaleY);

    std::vector<ShelfCandidate> shelves = findShelves(
        horizontalSegments,
        vpOnFullSize,
        image.size(),
        scaleX, scaleY,
        binSize, 
        minWeight
    );

    float mergeThresholdPixels = mergeThreshold * image.rows;
    std::vector<ShelfCandidate> finalShelves = mergeCloseShelves(shelves, mergeThresholdPixels);

    for (const auto& shelf : finalShelves) {
        cv::line(visa, shelf.leftPoint, shelf.rightPoint, cv::Scalar(0, 0, 255), 3);
    }

    if (!cv::imwrite(outputPath, visa)) {
        std::cerr << "Error: Could not save result to " << outputPath << "\n";
        return 1;
    }

    return 0;
}
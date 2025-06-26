#include <sdvprp.hpp> 
#include <nlohmann/json.hpp>

void evaluate(
    const std::string& gtJsonPath,
    const std::string& imageName,
    const std::vector<ShelfCandidate>& predictedShelves,
    const cv::Mat& originalImage,
    const std::string& outputPath,
    float thresholdPercent = 0.1f
) {
    std::ifstream file(gtJsonPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open GT JSON: " << gtJsonPath << std::endl;
        return;
    }

    nlohmann::json j;
    file >> j;

    const auto& regions = j[imageName]["regions"];
    std::vector<std::pair<cv::Point2f, cv::Point2f>> gtLines;
    for (const auto& region : regions) {
        const auto& shape = region["shape_attributes"];
        if (shape.contains("name") && shape["name"] == "polyline") {
            const auto& x = shape["all_points_x"];
            const auto& y = shape["all_points_y"];
            if (x.size() == 2 && y.size() == 2) {
                cv::Point2f p1(x[0].get<float>(), y[0].get<float>());
                cv::Point2f p2(x[1].get<float>(), y[1].get<float>());
                gtLines.emplace_back(p1, p2);
            }
        }
    }

    std::vector<int> usedPred(predictedShelves.size(), 0);
    std::vector<bool> matchedGT(gtLines.size(), false);
    std::vector<bool> isMatched(predictedShelves.size(), false);
    std::vector<bool> sh(predictedShelves.size(), false);
    float matchedLines = 0;
    for (size_t i = 0; i < gtLines.size(); ++i) {
        const auto& [g1, g2] = gtLines[i];
        float gtLength = cv::norm(g1 - g2);

        for (size_t j = 0; j < predictedShelves.size(); ++j) {
            const auto& pred = predictedShelves[j];
            cv::Point2f p1 = pred.leftPoint;
            cv::Point2f p2 = pred.rightPoint;

            float d1 = cv::norm(p1 - g1);
            float d2 = cv::norm(p2 - g2);
            float avgDist = 0.5f * (d1 + d2);
            float relDist = avgDist / gtLength;

            if (relDist <= thresholdPercent) {
                usedPred[j]++;
                isMatched[j] = true;
                matchedGT[i] = true;
                if (usedPred[j] <= 1 && sh[j] == false) {
                    sh[j] = true;

                }
            }
        }
    }
    for (size_t i = 0; i < matchedGT.size(); ++i) {
        if (matchedGT[i]) {
            matchedLines++;
        }
    }
    size_t gtShelves = gtLines.size() / 2;
    float TP = matchedLines / 2;
    float TPR = TP / gtShelves;

    float FN = static_cast<float>(gtShelves) - TP;
    int unmatchedPred = 0;
    for (bool matched : isMatched) {
        if (!matched) unmatchedPred++;
    }
    float FP = unmatchedPred / 2.0;
    float FPS = gtShelves > 0 ? static_cast<float>(FP / gtShelves) : 0;

    cv::Mat vis;
    originalImage.copyTo(vis);

    // GT (синие пунктиры)
    for (const auto& [p1, p2] : gtLines) {
        for (float alpha = 0; alpha <= 1.0; alpha += 0.05f) {
            cv::Point pt = p1 + alpha * (p2 - p1);
            cv::circle(vis, pt, 1, cv::Scalar(255, 0, 0), -1);
        }
    }

    for (size_t i = 0; i < predictedShelves.size(); ++i) {
        const auto& pred = predictedShelves[i];
        cv::Point p1 = pred.leftPoint;
        cv::Point p2 = pred.rightPoint;
        cv::Scalar color = isMatched[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255); // Зеленый для TP, красный для FP
        std::string label = isMatched[i] ? "TP" : "FP";

        cv::line(vis, p1, p2, color, 2);
        cv::putText(vis, label, 0.5f * (p1 + p2), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }

    cv::imwrite(outputPath, vis);

    std::cout << "=== Evaluation: " << imageName << " ===\n";
    std::cout << "GT shelves       : " << gtShelves << "\n";
    std::cout << "Predicted lines  : " << predictedShelves.size() << "\n";
    std::cout << "Matched lines    : " << matchedLines << "\n";
    std::cout << "TP               : " << std::fixed << std::setprecision(1) << TP << "\n";
    std::cout << "FN               : " << FN << "\n";
    std::cout << "FP               : " << FP << "\n";
    std::cout << "TPR              : " << std::setprecision(2) << TPR * 100.0f << "%\n";
    std::cout << "FPS              : " << FPS * 100.0f << "%\n";
    std::cout << "Saved to         : " << outputPath << "\n";
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
            << " path_to_image path_to_result "
            << "[lowThresh] [minWeight] [minLength] [mergeThreshold] [binSize]\n";
        return 1;
    }

    std::string imagePath = argv[1];
    std::string outputPath = argv[2];

    float lowThresh = 0.2f;
    float minWeight = 310.0f;
    float minLength = 70.0f;
    float mergeThreshold = 0.015f;
    float binSize = 2.0f;

    if (argc > 3) lowThresh = std::stof(argv[3]);
    if (argc > 4) minWeight = std::stof(argv[4]);
    if (argc > 5) minLength = std::stof(argv[5]);
    if (argc > 6) mergeThreshold = std::stof(argv[6]);
    if (argc > 7) binSize = std::stof(argv[7]);

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
    cv::Mat edgeMap = thresholdEdges(normalizedGrad, lowThresh);

    auto segments = followEdges(edgeMap, gradResult.gx, gradResult.gy);
    auto horizontalSegments = filterHorizontalSegments(segments, minLength);

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
    evaluate("C:\\Users\\vadim\\Pictures\\100.json", "100.jpg", finalShelves, image, "C:\\Users\\vadim\\Pictures\\est_100.png", 0.02f);



    return 0;
}

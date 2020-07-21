#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace {

class CannyApplier
{
    std::string m_win_name;
    cv::Mat m_src;
    cv::Mat m_src_gray;
    cv::Mat m_white;
    int m_threshold = 0;
    int m_ratio = 3;
    int m_apperture_size = 3;
    int m_blur_size = 1;
    int m_dilation_iter = 0;
    int m_erosion_iter = 0;

    static constexpr std::array s_apperture_size_values = {3, 5, 7};
    static constexpr std::array s_blur_size_values = {1, 3, 6, 8, 10, 13, 15, 18, 25};

public:
    static constexpr int max_threshold = 100;
    static constexpr int max_ratio = 50;
    static constexpr int max_apperture_size = s_apperture_size_values.size() - 1;
    static constexpr int max_blur_size = s_blur_size_values.size() - 1;
    static constexpr int max_dilation_iter = 10;
    static constexpr int max_erosion_iter = 10;

    static void apply_threshold(int pos, void * obj);
    static void apply_ratio(int pos, void * obj);
    static void apply_apperture(int pos, void * obj);
    static void apply_blur(int pos, void * obj);
    static void apply_dilation(int pos, void * obj);
    static void apply_erosion(int pos, void * obj);

    void initial() const
    {
        redraw();
    }

    CannyApplier(const char * win_name, const char * file_name)
        : m_win_name(win_name)
        , m_src(cv::imread(file_name, cv::IMREAD_COLOR))
        , m_white(m_src.size(), m_src.type())
    {
        if (!m_src.empty()) {
            cv::cvtColor(m_src, m_src_gray, cv::COLOR_BGR2GRAY);
            cv::normalize(m_src_gray, m_src_gray, 0, 1, cv::NORM_MINMAX);
            m_white = cv::Scalar::all(0xFF);
        }
    }

    bool empty() const
    { return m_src.empty(); }

private:
    auto make_canny() const
    {
        cv::Mat edges, dst;
        dst.create(m_src.size(), m_src.type());
        cv::blur(m_src_gray, edges, cv::Size(m_blur_size, m_blur_size));
        cv::Canny(edges, edges, m_threshold, m_threshold * m_ratio, m_apperture_size);
        const auto morpher1 = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
        const auto morpher2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
        if (m_dilation_iter > 0) {
            cv::dilate(edges, edges, morpher1, cv::Point(-1, -1), m_dilation_iter);
        }
        if (m_erosion_iter > 0) {
            cv::erode(edges, edges, morpher1, cv::Point(-1, -1), m_erosion_iter);
        }
        cv::dilate(edges, edges, morpher2, cv::Point(-1, -1), 5);
        cv::medianBlur(edges, edges, 5);

        cv::Mat nuclei = smooth_contours(edges);
        //cv::dilate(nuclei, nuclei, morpher2, cv::Point(-1, -1), 5);
        //cv::medianBlur(nuclei, nuclei, 5);
        dst = cv::Scalar::all(0);
        m_white.copyTo(dst, nuclei);
        return dst;
    }

    cv::Mat smooth_contours(const cv::Mat & src) const
    {
        std::vector<std::vector<cv::Point>> contours, approx_contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        approx_contours.resize(contours.size());
        for (std::size_t i = 0; i < contours.size(); ++i) {
            //cv::approxPolyDP(contours[i], approx_contours[i], 1, true);
            cv::convexHull(contours[i], approx_contours[i]);
        }

        cv::Mat dst;
        dst.create(src.size(), src.type());
        dst = cv::Scalar::all(0);
        cv::drawContours(dst, approx_contours, -1, cv::Scalar::all(255), cv::FILLED);
        return dst;
    }

    void redraw() const
    {
        cv::imshow(m_win_name, make_canny());
    }

#define UPDATER(field) \
    void update_##field(const int field) \
    { \
        if (field != m_##field) { \
            m_##field = field; \
            redraw(); \
        } \
    }
    UPDATER(threshold)
    UPDATER(ratio)
    UPDATER(apperture_size)
    UPDATER(blur_size)
    UPDATER(dilation_iter)
    UPDATER(erosion_iter)
#undef UPDATER
};

void CannyApplier::apply_threshold(const int pos, void * obj)
{
    auto self = static_cast<CannyApplier *>(obj);
    self->update_threshold(pos);
}

void CannyApplier::apply_ratio(const int pos, void * obj)
{
    auto self = static_cast<CannyApplier *>(obj);
    self->update_ratio(pos);
}

void CannyApplier::apply_apperture(const int pos, void * obj)
{
    auto self = static_cast<CannyApplier *>(obj);
    self->update_apperture_size(s_apperture_size_values[pos]);
}

void CannyApplier::apply_blur(const int pos, void * obj)
{
    auto self = static_cast<CannyApplier *>(obj);
    self->update_blur_size(s_blur_size_values[pos]);
}

void CannyApplier::apply_dilation(const int pos, void * obj)
{
    auto self = static_cast<CannyApplier *>(obj);
    self->update_dilation_iter(pos);
}

void CannyApplier::apply_erosion(const int pos, void * obj)
{
    auto self = static_cast<CannyApplier *>(obj);
    self->update_erosion_iter(pos);
}

} // anonymous namespace

int main(int argc, char ** argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <image filename>" << std::endl;
        return -1;
    }

    const char * win_name = "Edge Map";

    CannyApplier canny(win_name, argv[1]);
    if (canny.empty()) {
        std::cerr << "Could not open image " << argv[1] << std::endl;
        return -2;
    }

    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Min Threshold:", win_name, nullptr, CannyApplier::max_threshold, CannyApplier::apply_threshold, &canny);
    cv::createTrackbar("Threshold Ratio:", win_name, nullptr, CannyApplier::max_ratio, CannyApplier::apply_ratio, &canny);
    cv::createTrackbar("Apperture Size:", win_name, nullptr, CannyApplier::max_apperture_size, CannyApplier::apply_apperture, &canny);
    cv::createTrackbar("Blur Size:", win_name, nullptr, CannyApplier::max_blur_size, CannyApplier::apply_blur, &canny);
    cv::createTrackbar("Dilation Iters:", win_name, nullptr, CannyApplier::max_dilation_iter, CannyApplier::apply_dilation, &canny);
    cv::createTrackbar("Erosion Iters:", win_name, nullptr, CannyApplier::max_erosion_iter, CannyApplier::apply_erosion, &canny);

    std::cout << "Initial processing..." << std::endl;
    canny.initial();

    cv::waitKey(0);
}

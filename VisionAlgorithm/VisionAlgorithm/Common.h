#define PI  3.14159265358979323846
struct MatchResult {
    int point[8] = { -1,-1,-1,-1,-1,-1,-1,-1 };
    double angle;
    double score;
    int textlen = 0;
    char text[100];
};
struct BlobResult {
    int point[8];
    double area;
    double arclen;
    double ratio;
};

template<typename T>
struct TransPoint {
    T x;
    T y;
};
//struct TransPoint
//{
//    int x;
//    int y;
//};
struct GeometryData {
    double centerX;
    double centerY;
    double radiusX;
    double radiusY;
    double angle;
};

enum ReplacementMethod {
    NEIGHBOR_MIN,
    NEIGHBOR_MAX,
    NEIGHBOR_INTERPOLATION,
    GLOBAL
};

enum GlobalMethod {
    FIXED,
    IMAGE_MIN,
    IMAGE_MAX,
    IMAGE_AVERAGE
};

enum Direction {
    HORIZONTAL,
    VERTICAL,
    BOTH
};

enum Channel {
    GRAY,
    PLANE_0, // 红色通道
    PLANE_1, // 绿色通道
    PLANE_2  // 蓝色通道
};

enum FilterType { GAUSSIAN, MEAN, MEDIAN };

enum OverflowMode {
    WRAP,  // 封装
    CLAMP  // 箝位
};


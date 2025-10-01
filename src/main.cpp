#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

//  定义代价函数（残差计算）
struct ProjectileCostFunctor 
{
    double x_obs, y_obs, t_obs;  // 观测的 (x, y, t)

    ProjectileCostFunctor(double x, double y, double t) 
        : x_obs(x), y_obs(y), t_obs(t) {}

    template <typename T>
    bool operator()(const T* const vx,    // 初始x速度
                    const T* const vy,    // 初始y速度
                    const T* const g,     // 重力加速度
                    const T* const k,     // 空气阻力系数
                    T* residuals) const
    {
        T t = T(t_obs);
        T exp_kt = exp(-T(*k) * t);
        
        T x_pred = T(0) + (*vx) / (*k) * (T(1) - exp_kt);
        T y_pred = T(0) + (*vy + (*g) / (*k)) / (*k) * (T(1) - exp_kt) - (*g) / (*k) * t;  // 模型预测的x、y
       
        residuals[0] = x_obs - x_pred;
        residuals[1] = y_obs - y_pred;   // 残差计算：观测-预测
        return true;
    }
};

int main() 
{
    cv::VideoCapture cap("/home/f/CERE/video.mp4"); 
    if (!cap.isOpened()) 
    {
        std::cerr << "无法打开视频！" << std::endl;
        return -1;   // 读取视频
    }

    std::vector<cv::Point2f> points;  // 存储轨迹点 (x, y)
    std::vector<double> times;        // 存储对应时间 t
    double fps =60.0; 
    int frame_idx = 0;

    cv::Mat frame, gray_frame;
    std::vector<cv::Point2f> corners;

    while (cap.read(frame)) //从视频捕获对象 cap 中读取一帧图像到 frame 中
    {
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);// 转换为灰度图，用于角点检测
        cv::goodFeaturesToTrack(gray_frame, corners, 1, 0.01, 10.0 );  //检测角点
    
        if (!corners.empty()) 
        {
        double t = frame_idx / fps;  // 计算出当前帧对应的时间，用于后续结合弹丸位置进行轨迹分析等操作
        points.push_back(corners[0]);  //将当前帧检测到的弹丸位置添加到 std::vector<cv::Point2f> 容器中
        times.push_back(t); //将当前帧计算得到的时间 t 添加到 times 容器中
        }
        frame_idx++;  //读取下一帧
    }
    cap.release();

    //  构建 Ceres 优化问题
    ceres::Problem problem;
    double vx , vy ;
    double g = 300.0;               // 初始重力猜测
    double k = 0.1;                 // 初始空气阻力猜测

    if (points.size() >= 2) 
    {
    double dt = times[1] - times[0];
    double vx_init = (points[1].x - points[0].x) / dt;
    double vy_init = (points[1].y - points[0].y) / dt;
    vx = vx_init;
    vy = vy_init;
    }
    else 
    {
    vx = 100.0;  // 若点数不足，用场景合理值
    vy = 100.0;
    }

    // 为每个观测点添加代价函数
    for (size_t i = 0; i < points.size(); ++i) 
    {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ProjectileCostFunctor, 2, 1, 1, 1, 1>  //2：残差的维度（x、y 两个方向的残差)  1, 1, 1, 1：待优化参数的维度（vx​,vy​,g,k 各为 1 维）
           (new ProjectileCostFunctor(points[i].x, points[i].y, times[i]));  //传入当前观测点的 x,y 坐标和时间 t，用于计算残差
        problem.AddResidualBlock(
            cost_function,
             new ceres::HuberLoss(1.0),  // 对残差大于1.0的点降权
            &vx, &vy, &g, &k);
    }

    problem.SetParameterLowerBound(&g, 0, 100.0);
    problem.SetParameterUpperBound(&g, 0, 1000.0);
    problem.SetParameterLowerBound(&k, 0, 0.01);
    problem.SetParameterUpperBound(&k, 0, 1.0);        // 设置参数范围（g: 100-1000, k: 0.01-1）

    // 配置并运行优化器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  // 线性求解器
    options.max_num_iterations = 2000;          // 足够的迭代次数
    options.parameter_tolerance = 1e-12;        // 收紧参数收敛阈值
    options.function_tolerance = 1e-10;         // 收紧目标函数收敛阈值
    options.gradient_tolerance = 1e-12;         // 收紧梯度收敛阈值
    options.minimizer_progress_to_stdout = true;   // 打印优化过程

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;  //运行优化
    
    std::cout << "vx = " << vx << " px/s" << std::endl;
    std::cout << "vy = " << vy << " px/s" << std::endl;
    std::cout << "g = " << g << " px/s²" << std::endl;
    std::cout << "k = " << k << " 1/s" << std::endl;

    return 0;
}
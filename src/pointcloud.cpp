#include "pointcloud.h"
#include "simpleicp.h"

/**
 * @brief PointCloud 类的构造函数，初始化点云数据和选择状态
 * @param X 包含点的坐标矩阵 (n x 3)，每行表示一个点的 x, y, z 坐标
 */
PointCloud::PointCloud(Eigen::MatrixXd X) : X_{X}, sel_{std::vector<bool>(X.rows(), true)} {}

/**
 * @brief 获取所有被选中点的坐标矩阵
 * @return 包含被选中点的坐标矩阵 (m x 3)，m 为选中点的数量
 */
Eigen::MatrixXd PointCloud::GetXOfSelectedPts()
{
  auto sel_idx = GetIdxOfSelectedPts(); // 获取被选中点的索引
  Eigen::MatrixXd X_sel(sel_idx.size(), 3); // 创建结果矩阵，大小为选中点数 x 3
  for (uint i = 0; i < sel_idx.size(); i++)
  {
    X_sel(i, 0) = X_(sel_idx[i], 0); // 提取 x 坐标
    X_sel(i, 1) = X_(sel_idx[i], 1); // 提取 y 坐标
    X_sel(i, 2) = X_(sel_idx[i], 2); // 提取 z 坐标
  }
  return X_sel;
}

/**
 * @brief 获取被选中点的索引列表
 * @return 包含被选中点索引的向量
 */
std::vector<int> PointCloud::GetIdxOfSelectedPts()
{
  std::vector<int> idx; // 存储被选中点的索引
  for (uint i = 0; i < NoPts(); i++)
  {
    if (sel_[i]) // 检查是否为选中点
    {
      idx.push_back(i); // 添加索引到向量
    }
  }
  return idx;
}

/**
 * @brief 根据最大距离选择与给定点集重叠的点
 * @param X 另一个点云的坐标矩阵 (m x 3)
 * @param max_range 最大允许距离，超出此距离的点将被取消选中
 */
void PointCloud::SelectInRange(const Eigen::MatrixX3d &X, const double &max_range)
{
  auto sel_idx{GetIdxOfSelectedPts()}; // 获取当前选中点的索引
  auto no_selected_points = sel_idx.size(); // 记录选中点数量

  // 执行最近邻搜索
  auto X_query{GetXOfSelectedPts()}; // 获取选中点的坐标
  auto idx_nn{KnnSearch(X, X_query)}; // 查找每个查询点最近的邻居索引

  // 计算到最近邻点的距离
  Eigen::VectorXd dists(no_selected_points); // 存储距离
  dists.fill(NAN); // 初始化为 NaN

  for (uint i = 0; i < no_selected_points; i++)
  {
    double x_query{X_query(i, 0)}; // 查询点的 x 坐标
    double y_query{X_query(i, 1)}; // 查询点的 y 坐标
    double z_query{X_query(i, 2)}; // 查询点的 z 坐标

    double x_nn{X(idx_nn(i), 0)}; // 最近邻点的 x 坐标
    double y_nn{X(idx_nn(i), 1)}; // 最近邻点的 y 坐标
    double z_nn{X(idx_nn(i), 2)}; // 最近邻点的 z 坐标

    double dx{x_query - x_nn}; // x 方向差值
    double dy{y_query - y_nn}; // y 方向差值
    double dz{z_query - z_nn}; // z 方向差值

    double dist{sqrt(dx * dx + dy * dy + dz * dz)}; // 计算欧几里得距离

    dists(i) = dist; // 存储距离
  }

  // 取消超出最大距离的点的选中状态
  for (uint i = 0; i < no_selected_points; i++)
  {
    if (dists(i) > max_range) // 如果距离超过 max_range
    {
      sel_[sel_idx[i]] = false; // 取消该点的选中状态
    }
  }
}

/**
 * @brief 随机选择指定数量的点作为选中点
 * @param n 期望选择的点数
 */
void PointCloud::SelectNPts(const uint &n)
{
  auto sel_idx{GetIdxOfSelectedPts()}; // 获取当前选中点的索引

  if (n < sel_idx.size()) // 如果要求的选择数少于当前选中数
  {
    // 首先取消所有点的选中状态
    for (long i = 1; i < NoPts(); i++)
    {
      sel_[i] = false; // 取消选中
    }

    // 重新选择 n 个点
    auto idx_not_rounded{Eigen::VectorXd::LinSpaced(n, 0, static_cast<uint>(sel_idx.size()) - 1)}; // 生成线性间隔索引
    for (uint i = 0; i < n; i++)
    {
      uint idx_rounded{static_cast<uint>(round(idx_not_rounded(i)))}; // 四舍五入获取整数索引
      sel_[sel_idx[idx_rounded]] = true; // 重新选中点
    }
  }
}

/**
 * @brief 估计选中点的法向量和平面度
 * @param neighbors 用于计算法向量的邻居点数
 */
void PointCloud::EstimateNormals(const int &neighbors)
{
  // 初始化法向量和平面度向量，填充为 NaN
  nx_ = Eigen::VectorXd(NoPts());
  nx_.fill(NAN);
  ny_ = Eigen::VectorXd(NoPts());
  ny_.fill(NAN);
  nz_ = Eigen::VectorXd(NoPts());
  nz_.fill(NAN);
  planarity_ = Eigen::VectorXd(NoPts());
  planarity_.fill(NAN);

  Eigen::MatrixXi mat_idx_nn(X_.rows(), neighbors); // 存储每个点的邻居索引
  mat_idx_nn = KnnSearch(X_, GetXOfSelectedPts(), neighbors); // 执行最近邻搜索

  auto sel_idx = GetIdxOfSelectedPts(); // 获取选中点的索引
  for (uint i = 0; i < sel_idx.size(); i++)
  {
    // 构建邻居点的坐标矩阵
    Eigen::MatrixXd X_nn(neighbors, 3);
    for (int j = 0; j < neighbors; j++)
    {
      X_nn(j, 0) = X_(mat_idx_nn(i, j), 0); // x 坐标
      X_nn(j, 1) = X_(mat_idx_nn(i, j), 1); // y 坐标
      X_nn(j, 2) = X_(mat_idx_nn(i, j), 2); // z 坐标
    }

    // 计算协方差矩阵
    Eigen::MatrixXd centered = X_nn.rowwise() - X_nn.colwise().mean(); // 中心化数据
    Eigen::MatrixXd C = (centered.adjoint() * centered) / double(X_nn.rows() - 1); // 协方差矩阵

    // 使用特征值分解求解法向量（最小特征值对应的特征向量）
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C); // 快速求解对称矩阵的特征值和特征向量
    auto eigenvectors{es.eigenvectors()}; // 特征向量
    auto eigenvalues{es.eigenvalues()};   // 特征值（升序排列）
    nx_[sel_idx[i]] = eigenvectors(0, 0); // x 方向法向量分量
    ny_[sel_idx[i]] = eigenvectors(1, 0); // y 方向法向量分量
    nz_[sel_idx[i]] = eigenvectors(2, 0); // z 方向法向量分量
    planarity_[sel_idx[i]] = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]; // 计算平面度
  }
}

/**
 * @brief 使用给定的变换矩阵转换点云坐标
 * @param H 4x4 变换矩阵，包含旋转和平移
 */
void PointCloud::Transform(Eigen::Matrix<double, 4, 4> &H)
{
  X_ = (H * X_.transpose().colwise().homogeneous()).topRows<3>().transpose(); // 应用变换
}

/**
 * @brief 获取点云中点的总数
 * @return 点云中点的数量
 */
long PointCloud::NoPts() { return X_.rows(); }

// Getters
/**
 * @brief 获取点云的坐标矩阵
 * @return 包含所有点的坐标矩阵 (n x 3)
 */
const Eigen::MatrixXd &PointCloud::X() { return X_; }

/**
 * @brief 获取 x 方向的法向量分量
 * @return x 方向法向量向量
 */
const Eigen::VectorXd &PointCloud::nx() { return nx_; }

/**
 * @brief 获取 y 方向的法向量分量
 * @return y 方向法向量向量
 */
const Eigen::VectorXd &PointCloud::ny() { return ny_; }

/**
 * @brief 获取 z 方向的法向量分量
 * @return z 方向法向量向量
 */
const Eigen::VectorXd &PointCloud::nz() { return nz_; }

/**
 * @brief 获取每个点的平面度值
 * @return 平面度向量
 */
const Eigen::VectorXd &PointCloud::planarity() { return planarity_; }

/**
 * @brief 获取点的选中状态向量
 * @return 布尔向量，表示每个点的选中状态
 */
const std::vector<bool> &PointCloud::sel() { return sel_; }
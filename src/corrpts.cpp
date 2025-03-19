#include "corrpts.h"
#include "simpleicp.h"

/**
 * @brief CorrPts 类的构造函数，初始化固定点云和移动点云
 * @param pcF 固定点云 (PointCloud 对象)
 * @param pcM 移动点云 (PointCloud 对象)
 */
CorrPts::CorrPts(PointCloud &pcF, PointCloud &pcM) : pcF_{pcF}, pcM_{pcM} {}

/**
 * @brief 匹配固定点云和移动点云中的最近点对
 *        为固定点云的每个选中点在移动点云中找到最近邻点
 */
void CorrPts::Match()
{
  auto X_sel_pcF = pcF_.GetXOfSelectedPts(); // 获取固定点云中选中点的坐标
  auto X_sel_pcM = pcM_.GetXOfSelectedPts(); // 获取移动点云中选中点的坐标

  idx_pcF_ = pcF_.GetIdxOfSelectedPts(); // 获取固定点云中选中点的索引
  idx_pcM_ = std::vector<int>(idx_pcF_.size()); // 初始化移动点云的对应点索引向量

  Eigen::MatrixXi mat_idx_nn(idx_pcF_.size(), 1); // 存储最近邻点的索引
  mat_idx_nn = KnnSearch(X_sel_pcM, X_sel_pcF, 1); // 在移动点云中为固定点云的每个点找最近邻点
  for (int i = 0; i < mat_idx_nn.rows(); i++)
  {
    idx_pcM_[i] = mat_idx_nn(i, 0); // 记录移动点云中对应点的索引
  }

  GetPlanarityFrompcF(); // 获取固定点云中选中点的平面度
  ComputeDists(); // 计算对应点之间的距离
}

/**
 * @brief 从固定点云中提取选中点的平面度
 */
void CorrPts::GetPlanarityFrompcF()
{
  planarity_ = Eigen::VectorXd(idx_pcF_.size()); // 初始化平面度向量
  for (uint i = 0; i < idx_pcF_.size(); i++)
  {
    planarity_[i] = pcF_.planarity()[idx_pcF_[i]]; // 提取固定点云中对应点的平面度
  }
}

/**
 * @brief 计算对应点对之间的点到平面距离
 *        距离定义为移动点到固定点所在平面的距离（沿法向量方向）
 */
void CorrPts::ComputeDists()
{
  dists_ = Eigen::VectorXd(idx_pcF_.size()); // 初始化距离向量
  dists_.fill(NAN); // 填充为 NaN

  for (uint i = 0; i < idx_pcF_.size(); i++)
  {
    // 获取固定点云中点的坐标
    double x_pcF = pcF_.X()(idx_pcF_[i], 0);
    double y_pcF = pcF_.X()(idx_pcF_[i], 1);
    double z_pcF = pcF_.X()(idx_pcF_[i], 2);

    // 获取移动点云中对应点的坐标
    double x_pcM = pcM_.X()(idx_pcM_[i], 0);
    double y_pcM = pcM_.X()(idx_pcM_[i], 1);
    double z_pcM = pcM_.X()(idx_pcM_[i], 2);

    // 获取固定点云中点的法向量
    double nx_pcF = pcF_.nx()(idx_pcF_[i]);
    double ny_pcF = pcF_.ny()(idx_pcF_[i]);
    double nz_pcF = pcF_.nz()(idx_pcF_[i]);

    // 计算点到平面的距离：(pM - pF) 点积 nF
    double dist{(x_pcM - x_pcF) * nx_pcF + (y_pcM - y_pcF) * ny_pcF + (z_pcM - z_pcF) * nz_pcF};

    dists_(i) = dist; // 存储距离
  }
}

/**
 * @brief 拒绝不满足条件的对应点对
 *        基于距离的统计特性（中值和 MAD）和最小平面度进行筛选
 * @param min_planarity 平面度阈值，低于此值的对应点对将被剔除
 */
void CorrPts::Reject(const double &min_planarity)
{
  auto med{Median(dists_)}; // 计算距离的中值
  auto sigmad{1.4826 * MAD(dists_)}; // 计算距离的 MAD（中值绝对偏差），并缩放为标准差估计
  std::vector<bool> keep(dists_.size(), true); // 标记哪些对应点对需要保留
  for (int i = 0; i < dists_.size(); i++)
  {
    // 剔除条件：距离偏离中值超过 3 倍标准差，或平面度低于阈值
    if ((abs(dists_[i] - med) > 3 * sigmad) | (planarity_[i] < min_planarity))
    {
      keep[i] = false; // 标记为不保留
    }
  }

  // 计算保留的对应点数量
  size_t no_remaining_pts = count(keep.begin(), keep.end(), true);

  // 创建新的索引和距离向量，仅保留符合条件的对应点
  std::vector<int> idx_pcF_new(no_remaining_pts);
  std::vector<int> idx_pcM_new(no_remaining_pts);
  Eigen::VectorXd dists_new(no_remaining_pts);
  int j{0};
  for (int i = 0; i < dists_.size(); i++)
  {
    if (keep[i])
    {
      idx_pcF_new[j] = idx_pcF_[i]; // 保留固定点云的索引
      idx_pcM_new[j] = idx_pcM_[i]; // 保留移动点云的索引
      dists_new[j] = dists_[i]; // 保留距离
      j++;
    }
  }

  // 更新成员变量
  idx_pcF_ = idx_pcF_new;
  idx_pcM_ = idx_pcM_new;
  dists_ = dists_new;
}

/**
 * @brief 根据欧拉角生成旋转矩阵
 * @param alpha1 绕 x 轴的旋转角度（弧度）
 * @param alpha2 绕 y 轴的旋转角度（弧度）
 * @param alpha3 绕 z 轴的旋转角度（弧度）
 * @return 3x3 旋转矩阵
 */
Eigen::Matrix3d EulerAnglesToRotationMatrix(float alpha1, float alpha2, float alpha3)
{
  Eigen::Matrix3d R;
  R << std::cos(alpha2) * std::cos(alpha3),
       -std::cos(alpha2) * std::sin(alpha3),
       std::sin(alpha2),

       std::cos(alpha1) * std::sin(alpha3) + std::sin(alpha1) * std::sin(alpha2) * std::cos(alpha3),
       std::cos(alpha1) * std::cos(alpha3) - std::sin(alpha1) * std::sin(alpha2) * std::sin(alpha3),
       -std::sin(alpha1) * std::cos(alpha2),

       std::sin(alpha1) * std::sin(alpha3) - std::cos(alpha1) * std::sin(alpha2) * std::cos(alpha3),
       std::sin(alpha1) * std::cos(alpha3) + std::cos(alpha1) * std::sin(alpha2) * std::sin(alpha3),
       std::cos(alpha1) * std::cos(alpha2);

  return R;
}

/**
 * @brief 估计刚性变换矩阵（旋转 + 平移）
 * @param H 输出参数，4x4 刚性变换矩阵
 * @param residuals 输出参数，残差向量（变换后的点到平面距离）
 */
void CorrPts::EstimateRigidBodyTransformation(Eigen::Matrix<double, 4, 4> &H,
                                              Eigen::VectorXd &residuals)
{
  auto no_corr_pts{idx_pcF_.size()}; // 对应点对数量

  Eigen::MatrixXd A(no_corr_pts, 6); // 设计矩阵，用于最小二乘法
  Eigen::VectorXd l(no_corr_pts); // 观测向量

  // 构建线性方程组 A * x = l
  for (uint i = 0; i < no_corr_pts; i++)
  {
    // 固定点云中点的坐标
    double x_pcF = pcF_.X()(idx_pcF_[i], 0);
    double y_pcF = pcF_.X()(idx_pcF_[i], 1);
    double z_pcF = pcF_.X()(idx_pcF_[i], 2);

    // 移动点云中对应点的坐标
    double x_pcM = pcM_.X()(idx_pcM_[i], 0);
    double y_pcM = pcM_.X()(idx_pcM_[i], 1);
    double z_pcM = pcM_.X()(idx_pcM_[i], 2);

    // 固定点云中点的法向量
    double nx_pcF = pcF_.nx()(idx_pcF_[i]);
    double ny_pcF = pcF_.ny()(idx_pcF_[i]);
    double nz_pcF = pcF_.nz()(idx_pcF_[i]);

    // 构建设计矩阵 A 的第 i 行（基于点到平面距离约束）
    A(i, 0) = -z_pcM * ny_pcF + y_pcM * nz_pcF; // 旋转分量 (alpha1)
    A(i, 1) = z_pcM * nx_pcF - x_pcM * nz_pcF;  // 旋转分量 (alpha2)
    A(i, 2) = -y_pcM * nx_pcF + x_pcM * ny_pcF; // 旋转分量 (alpha3)
    A(i, 3) = nx_pcF; // 平移分量 (tx)
    A(i, 4) = ny_pcF; // 平移分量 (ty)
    A(i, 5) = nz_pcF; // 平移分量 (tz)

    // 观测值：点到平面距离
    l(i) = nx_pcF * (x_pcF - x_pcM) + ny_pcF * (y_pcF - y_pcM) + nz_pcF * (z_pcF - z_pcM);
  }

  // 使用 SVD 求解线性方程组 A * x = l，得到变换参数
  Eigen::Matrix<double, 6, 1> x{A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(l)};

  // 提取旋转和平移参数
  double alpha1{x(0)}; // 绕 x 轴旋转角度
  double alpha2{x(1)}; // 绕 y 轴旋转角度
  double alpha3{x(2)}; // 绕 z 轴旋转角度
  double tx{x(3)};     // x 方向平移
  double ty{x(4)};     // y 方向平移
  double tz{x(5)};     // z 方向平移

  // 构建变换矩阵 H
  H = Eigen::Matrix4d::Identity(); // 初始化为单位矩阵
  H.topLeftCorner<3, 3>() = EulerAnglesToRotationMatrix(alpha1, alpha2, alpha3); // 设置旋转部分
  H.topRightCorner<3, 1>() << tx, ty, tz; // 设置平移部分

  // 计算残差（变换后的点到平面距离）
  residuals = A * x - l;
}

// Getters
/**
 * @brief 获取固定点云
 * @return 固定点云对象
 */
const PointCloud &CorrPts::pcF() { return pcF_; }

/**
 * @brief 获取移动点云
 * @return 移动点云对象
 */
const PointCloud &CorrPts::pcM() { return pcM_; }

/**
 * @brief 获取固定点云中对应点的索引
 * @return 固定点云中对应点索引向量
 */
const std::vector<int> &CorrPts::idx_pcF() { return idx_pcF_; }

/**
 * @brief 获取移动点云中对应点的索引
 * @return 移动点云中对应点索引向量
 */
const std::vector<int> &CorrPts::idx_pcM() { return idx_pcM_; }

/**
 * @brief 获取对应点对的点到平面距离
 * @return 点到平面距离向量
 */
const Eigen::VectorXd &CorrPts::dists() { return dists_; }
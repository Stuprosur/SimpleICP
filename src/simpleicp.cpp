#include "simpleicp.h"
#include "corrpts.h"
#include "nanoflann.hpp"
#include "pointcloud.h"

const int LEAF_SIZE{200};

// Eigen::Matrix<double, 4, 4> SimpleICP(const Eigen::MatrixXd &X_fix,
//                                       const Eigen::MatrixXd &X_mov,
//                                       const int &correspondences,
//                                       const int &neighbors,
//                                       const double &min_planarity,
//                                       const double &max_overlap_distance,
//                                       const double &min_change,
//                                       const int &max_iterations)
// {
//   auto start = std::chrono::system_clock::now();

//   printf("Create point cloud objects ...\n");
//   PointCloud pc_fix{X_fix};
//   PointCloud pc_mov{X_mov};

//   if (max_overlap_distance > 0)
//   {
//     printf("Consider partial overlap of point clouds ...\n");
//     pc_fix.SelectInRange(pc_mov.X(), max_overlap_distance);
//     if (pc_fix.GetIdxOfSelectedPts().size() == 0)
//     {
//       char buff[200];
//       snprintf(buff, sizeof(buff),
//                "Point clouds do not overlap within max_overlap_distance = %.5f. "
//                "Consider increasing the value of max_overlap_distance.\n",
//                max_overlap_distance);
//       std::string error_msg{buff};
//       throw std::runtime_error(error_msg);
//     }
//   }

//   printf("Select points for correspondences in fixed point cloud ...\n");
//   pc_fix.SelectNPts(correspondences);

//   printf("Estimate normals of selected points ...\n");
//   pc_fix.EstimateNormals(neighbors);

//   // Initialization
//   Eigen::Matrix<double, 4, 4> H_old{Eigen::Matrix<double, 4, 4>::Identity()};
//   Eigen::Matrix<double, 4, 4> H_new;
//   Eigen::Matrix<double, 4, 4> dH;
//   Eigen::VectorXd residual_dists;
//   std::vector<double> residual_dists_mean;
//   std::vector<double> residual_dists_std;

//   printf("Start iterations ...\n");
//   for (int i = 0; i < max_iterations; i++)
//   {
//     CorrPts cp = CorrPts(pc_fix, pc_mov);

//     cp.Match();
//     cp.Reject(min_planarity);
//     auto initial_dists{cp.dists()};

//     cp.EstimateRigidBodyTransformation(dH, residual_dists);

//     pc_mov.Transform(dH);

//     H_new = H_old * dH;
//     H_old = H_new;

//     residual_dists_mean.push_back(residual_dists.mean());
//     residual_dists_std.push_back(Std(residual_dists));

//     if (i > 0)
//     {
//       if (CheckConvergenceCriteria(residual_dists_mean, residual_dists_std, min_change))
//       {
//         printf("Convergence criteria fulfilled -> stop iteration!\n");
//         break;
//       }
//     }

//     if (i == 0)
//     {
//       printf("%9s | %15s | %15s | %15s\n",
//              "Iteration",
//              "correspondences",
//              "mean(residuals)",
//              "std(residuals)");
//       printf("%9s | %15d | %15.4f | %15.4f\n",
//              "orig:0",
//              int(initial_dists.size()),
//              initial_dists.mean(),
//              Std(initial_dists));
//     }
//     printf("%9d | %15d | %15.4f | %15.4f\n",
//            i + 1,
//            int(residual_dists.size()),
//            residual_dists_mean.back(),
//            residual_dists_std.back());
//   }

//   printf("Estimated transformation matrix H:\n");
//   printf("[%12.6f %12.6f %12.6f %12.6f]\n",
//          H_new(0, 0),
//          H_new(0, 1),
//          H_new(0, 2),
//          H_new(0, 3));
//   printf("[%12.6f %12.6f %12.6f %12.6f]\n",
//          H_new(1, 0),
//          H_new(1, 1),
//          H_new(1, 2),
//          H_new(1, 3));
//   printf("[%12.6f %12.6f %12.6f %12.6f]\n",
//          H_new(2, 0),
//          H_new(2, 1),
//          H_new(2, 2),
//          H_new(2, 3));
//   printf("[%12.6f %12.6f %12.6f %12.6f]\n",
//          H_new(3, 0),
//          H_new(3, 1),
//          H_new(3, 2),
//          H_new(3, 3));

//   auto end = std::chrono::system_clock::now();
//   std::chrono::duration<double> elapsed_seconds = end - start;
//   printf("Finished in %.3f seconds!\n", elapsed_seconds.count());

//   return H_new;
// }

/**
 * @brief 使用简化的ICP算法将两个点云对齐，估计移动点云到固定点云的刚性变换矩阵
 * @param X_fix 固定点云的坐标矩阵 (n x 3)，每行是一个点的x, y, z坐标
 * @param X_mov 移动点云的坐标矩阵 (m x 3)，每行是一个点的x, y, z坐标
 * @param correspondences 在固定点云中选择用于匹配的点数
 * @param neighbors 计算法向量时考虑的邻居点数
 * @param min_planarity 拒绝匹配时使用的平面度阈值，低于此值的匹配被剔除
 * @param max_overlap_distance 点云重叠的最大距离，若>0则考虑部分重叠
 * @param min_change 收敛判断的最小变化阈值
 * @param max_iterations 最大迭代次数
 * @return 估计的4x4刚性变换矩阵H，将X_mov变换到X_fix的坐标系
 */
Eigen::Matrix<double, 4, 4> SimpleICP(const Eigen::MatrixXd &X_fix,
                                      const Eigen::MatrixXd &X_mov,
                                      const int &correspondences,
                                      const int &neighbors,
                                      const double &min_planarity,
                                      const double &max_overlap_distance,
                                      const double &min_change,
                                      const int &max_iterations)
{
  auto start = std::chrono::system_clock::now(); // 记录算法开始时间

  printf("Create point cloud objects ...\n");
  // 创建固定点云对象 【点云坐标矩阵X_、bool向量sel_】
  PointCloud pc_fix{X_fix};  
  // 创建移动点云对象 【点云坐标矩阵X_、bool向量sel_】
  PointCloud pc_mov{X_mov};  

  // 处理点云的部分重叠情况
  if (max_overlap_distance > 0)
  {
    printf("Consider partial overlap of point clouds ...\n");
    // 在固定点云中选择与移动点云距离小于max_overlap_distance的点 【bool向量sel_】
    pc_fix.SelectInRange(pc_mov.X(), max_overlap_distance); 
    if (pc_fix.GetIdxOfSelectedPts().size() == 0) // 如果没有重叠点，抛出异常
    {
      char buff[200];
      snprintf(buff, sizeof(buff),
               "Point clouds do not overlap within max_overlap_distance = %.5f. "
               "Consider increasing the value of max_overlap_distance.\n",
               max_overlap_distance);
      std::string error_msg{buff};
      throw std::runtime_error(error_msg);
    }
  }

  printf("Select points for correspondences in fixed point cloud ...\n");
  pc_fix.SelectNPts(correspondences); // 在固定点云中随机选择指定数量的点用于匹配 【bool向量sel_】

  printf("Estimate normals of selected points ...\n");
  pc_fix.EstimateNormals(neighbors); // 为选中的点估计法向量，使用指定数量的邻居点 【n_x, n_y, n_z, planarity_】

  // ---------------------------------------------Registration-----------------------------------------------
  // 初始化变换矩阵和相关变量
  Eigen::Matrix<double, 4, 4> H_old{Eigen::Matrix<double, 4, 4>::Identity()}; // 上一次的变换矩阵，初始为单位矩阵
  Eigen::Matrix<double, 4, 4> H_new; // 当前迭代的变换矩阵
  Eigen::Matrix<double, 4, 4> dH;    // 每次迭代的增量变换矩阵
  Eigen::VectorXd residual_dists;    // 残差距离向量
  std::vector<double> residual_dists_mean; // 每次迭代残差均值的记录
  std::vector<double> residual_dists_std;  // 每次迭代残差标准差的记录

  printf("Start iterations ...\n");
  for (int i = 0; i < max_iterations; i++) // 开始迭代对齐过程
  {
    CorrPts cp = CorrPts(pc_fix, pc_mov); // 创建对应点匹配对象

    cp.Match(); // 在移动点云中寻找与固定点云的最近点对应
    cp.Reject(min_planarity); // 拒绝平面度低于阈值的匹配对
    auto initial_dists{cp.dists()}; // 获取初始匹配的距离

    cp.EstimateRigidBodyTransformation(dH, residual_dists); // 估计刚性变换矩阵dH，并更新残差距离

    pc_mov.Transform(dH); // 对移动点云应用增量变换

    H_new = H_old * dH; // 更新总变换矩阵
    H_old = H_new;      // 保存当前变换作为下一次迭代的初始值

    residual_dists_mean.push_back(residual_dists.mean()); // 记录残差均值
    residual_dists_std.push_back(Std(residual_dists));   // 记录残差标准差

    // 检查收敛条件（从第二次迭代开始）
    if (i > 0)
    {
      if (CheckConvergenceCriteria(residual_dists_mean, residual_dists_std, min_change))
      {
        printf("Convergence criteria fulfilled -> stop iteration!\n");
        break; // 如果满足收敛条件，提前终止迭代
      }
    }

    // 输出迭代信息
    if (i == 0)
    {
      // 打印表头和初始状态
      printf("%9s | %15s | %15s | %15s\n",
             "Iteration",
             "correspondences",
             "mean(residuals)",
             "std(residuals)");
      printf("%9s | %15d | %15.4f | %15.4f\n",
             "orig:0",
             int(initial_dists.size()),
             initial_dists.mean(),
             Std(initial_dists));
    }
    printf("%9d | %15d | %15.4f | %15.4f\n",
           i + 1,
           int(residual_dists.size()),
           residual_dists_mean.back(),
           residual_dists_std.back());
  }

  // 打印最终估计的变换矩阵
  printf("Estimated transformation matrix H:\n");
  printf("[%12.6f %12.6f %12.6f %12.6f]\n",
         H_new(0, 0), H_new(0, 1), H_new(0, 2), H_new(0, 3));
  printf("[%12.6f %12.6f %12.6f %12.6f]\n",
         H_new(1, 0), H_new(1, 1), H_new(1, 2), H_new(1, 3));
  printf("[%12.6f %12.6f %12.6f %12.6f]\n",
         H_new(2, 0), H_new(2, 1), H_new(2, 2), H_new(2, 3));
  printf("[%12.6f %12.6f %12.6f %12.6f]\n",
         H_new(3, 0), H_new(3, 1), H_new(3, 2), H_new(3, 3));

  // 计算并打印运行时间
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  printf("Finished in %.3f seconds!\n", elapsed_seconds.count());

  return H_new; // 返回最终的变换矩阵
}



Eigen::MatrixXi KnnSearch(const Eigen::MatrixXd &X, const Eigen::MatrixXd &X_query, const int &k)
{
  // Create kd tree
  typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>
      kd_tree;
  kd_tree mat_index(3, std::cref(X), LEAF_SIZE);

  // Iterate over all query points
  Eigen::MatrixXi mat_idx_nn(X_query.rows(), k);
  for (int i = 0; i < X_query.rows(); i++)
  {
    // Query point
    std::vector<double> qp{X_query(i, 0), X_query(i, 1), X_query(i, 2)};

    // Search for nn of query point
    std::vector<size_t> idx_nn(k);
    std::vector<double> dists_nn(k); // not used
    nanoflann::KNNResultSet<double> resultSet(k);
    resultSet.init(&idx_nn[0], &dists_nn[0]);
    // mat_index.index->findNeighbors(resultSet, &qp[0], nanoflann::SearchParams(10));
    mat_index.index_->findNeighbors(resultSet, &qp[0], nanoflann::SearchParameters(10));
    

    // Save indices of nn to matrix
    for (int j = 0; j < k; j++)
    {
      mat_idx_nn(i, j) = idx_nn[j];
    }
  }
  return mat_idx_nn;
}

double Median(const Eigen::VectorXd &v)
{
  // VectorXd --> vector<double>
  std::vector<double> vv(v.size());
  for (int i = 1; i < v.size(); i++)
  {
    vv[i] = v[i];
  }

  // Median
  const auto median_it = vv.begin() + vv.size() / 2;
  nth_element(vv.begin(), median_it, vv.end());
  auto median = *median_it;

  return median;
}

double MAD(const Eigen::VectorXd &v)
{
  auto med{Median(v)};
  Eigen::VectorXd dmed(v.size());
  for (int i = 1; i < v.size(); i++)
  {
    dmed[i] = abs(v[i] - med);
  }
  auto mad{Median(dmed)};
  return mad;
}

double Std(const Eigen::VectorXd &v)
{
  double std{sqrt((v.array() - v.mean()).square().sum() / (v.size() - 1))};
  return std;
}

double Change(const double &new_val, const double &old_val)
{
  return abs((new_val - old_val) / old_val * 100);
}

bool CheckConvergenceCriteria(const std::vector<double> &residual_dists_mean,
                              const std::vector<double> &residual_dists_std,
                              const double &min_change)
{
  if (Change(residual_dists_mean.end()[-1], residual_dists_mean.end()[-2]) < min_change)
  {
    if (Change(residual_dists_std.end()[-1], residual_dists_std.end()[-2]) < min_change)
    {
      return true;
    }
  }
  return false;
}

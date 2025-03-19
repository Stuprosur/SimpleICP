#ifndef RUN_SIMPLEICP_CORRPTS_H
#define RUN_SIMPLEICP_CORRPTS_H

#include "pointcloud.h"

class CorrPts
{
public:
  CorrPts(PointCloud &pcF, PointCloud &pcM);

  // Matching of each selected point of pcF --> nn of selected points of pcM
  void Match();

  void GetPlanarityFrompcF();

  void ComputeDists();

  void Reject(const double &min_planarity);

  void EstimateRigidBodyTransformation(Eigen::Matrix<double, 4, 4> &H, Eigen::VectorXd &residuals);

  // Getters
  const PointCloud &pcF();
  const PointCloud &pcM();
  const std::vector<int> &idx_pcF();
  const std::vector<int> &idx_pcM();
  const Eigen::VectorXd &dists();
  const Eigen::VectorXd &planarity();

private:
  PointCloud pcF_;
  PointCloud pcM_;
  std::vector<int> idx_pcF_;
  std::vector<int> idx_pcM_;
  Eigen::VectorXd dists_;
  Eigen::VectorXd planarity_;
};

#endif // RUN_SIMPLEICP_CORRPTS_H

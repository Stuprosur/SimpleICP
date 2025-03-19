#include <iostream>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

void pcd_to_xyz(const std::string& pcd_file, const std::string& xyz_file) {
	// 创建点云对象
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// 加载 PCD 文件
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *cloud) == -1) {
		std::cerr << "错误：无法加载 PCD 文件 " << pcd_file << std::endl;
		return;
	}

	// 打开 XYZ 文件以写入
	std::ofstream out_file(xyz_file);
	if (!out_file.is_open()) {
		std::cerr << "错误：无法创建文件 " << xyz_file << std::endl;
		return;
	}

	// 遍历点云并写入 XYZ 文件
	for (const auto& point : cloud->points) {
		out_file << point.x << " " << point.y << " " << point.z << "\n";
	}

	std::cout << "成功将 " << pcd_file << " 转换为 " << xyz_file << std::endl;

	// 关闭文件
	out_file.close();
}

int main(int argc, char* argv[]) {
	// 检查命令行参数
	if (argc != 3) {
		std::cerr << "用法: " << argv[0] << " <input.pcd> <output.xyz>" << std::endl;
		return 1;
	}

	std::string pcd_file = argv[1];
	std::string xyz_file = argv[2];

	// 执行转换
	pcd_to_xyz(pcd_file, xyz_file);

	return 0;
}
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <limits>
#include <vector>
#include <Eigen/Core>

//ROS
#include <ros/ros.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <depth_deco/ROI_identifier.h>
#include "DEOpt/DEOpt.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree.h>

#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>

#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/visualization/range_image_visualizer.h>
 #include <pcl/visualization/cloud_viewer.h>
 #include <pcl/features/normal_3d.h>

#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/keypoints/sift_keypoint.h>

using namespace cv;
using namespace std;

//SIFT Keypoints parameters
const float min_scale = 0.01f; //Desviación estándar
const int n_octaves = 3; //Número de divisiones
const int n_scales_per_octave = 4; //Número de difuminaciones por octava
const float min_contrast = 0.001f;

//Sample Consensus Initial Alignment
const float min_sample_dist = 0.025f;
const float max_correspondence_dist = 0.01f;
const int nr_iters = 500;

//ICP parameters
const float max_correspondence_distance = 0.05f;
const float outlier_rejection_threshold = 0.05f;
const float transformation_epsilon = 0;
const int max_iterations = 100;

Eigen::Matrix4f computeInitialAlignment (const pcl::PointCloud<pcl::PointWithScale>::Ptr source_key, const pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features, const pcl::PointCloud<pcl::PointWithScale>::Ptr target_key, const pcl::PointCloud<pcl::FPFHSignature33>::Ptr tar_features){

  cout << "Initial Alignment" << endl;
  pcl::SampleConsensusInitialAlignment <pcl::PointWithScale, pcl::PointWithScale, pcl::FPFHSignature33> sac_ia;

  sac_ia.setMinSampleDistance (min_sample_dist);
  sac_ia.setMaxCorrespondenceDistance (max_correspondence_dist);
  sac_ia.setMaximumIterations (nr_iters);

  sac_ia.setInputSource (source_key);
  sac_ia.setSourceFeatures (src_features);

  sac_ia.setInputTarget (target_key);
  sac_ia.setTargetFeatures (tar_features);

  pcl::PointCloud<pcl::PointWithScale> registration_output;

  sac_ia.align (registration_output);

  cout << "Fitness Score: " << sac_ia.getFitnessScore() << endl;

  return (sac_ia.getFinalTransformation ());

}

Eigen::Matrix4f refine_icp (const pcl::PointCloud<pcl::PointXYZ>::Ptr source_points, const pcl::PointCloud<pcl::PointXYZ>::Ptr target_points, const Eigen::Matrix4f initial_alignment){

    cout << "ICP" << endl;
  	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaxCorrespondenceDistance (max_correspondence_distance);
    icp.setRANSACOutlierRejectionThreshold (outlier_rejection_threshold);
    icp.setTransformationEpsilon (transformation_epsilon);
    icp.setMaximumIterations (max_iterations);

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_points_transformed (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*source_points, *source_points_transformed, initial_alignment);

    icp.setInputSource (source_points_transformed);
    icp.setInputTarget (target_points);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_icp (new pcl::PointCloud<pcl::PointXYZ>); //ICP output pointcloud
    icp.align (*cloud_icp);

    cout << "ICP score: " << icp.getFitnessScore () << endl;

    return (icp.getFinalTransformation () * initial_alignment);
}

int main (int argc, char **argv){

  ros::init (argc, argv, "pair");

  cout << endl;
  cout << "Cargando los archivos..." << endl;
  cout << "---------------------------" << endl;
  // Object for storing the point cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);

  // Read a PCD file from disk.
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/diego/PIVOTEO/1.pcd", *source_cloud) != 0)
  {
    return -1;
  }
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/diego/PIVOTEO/50.pcd", *target_cloud) != 0)
  {
    return -1;
  }

  vector<int> nan_idx;
  pcl::removeNaNFromPointCloud (*source_cloud,*source_cloud,nan_idx);
  pcl::removeNaNFromPointCloud (*target_cloud,*target_cloud,nan_idx);
//Para ahorrar tiempo
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.setInputCloud(source_cloud);
  sor.filter(*source_cloud);
  sor.setInputCloud(target_cloud);
  sor.filter(*target_cloud);

  //Eliminar NaNs para icp



  //Cloud normals source
  cout << "Computing source cloud normals" << endl;
  pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
  pcl::PointCloud<pcl::PointNormal>::Ptr src_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz (new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setInputCloud(source_cloud);
  ne.setSearchMethod(tree_xyz);
  ne.setRadiusSearch(0.05);
  ne.compute(*src_normals);
  for(size_t i = 0;  i < src_normals->points.size(); ++i) {
      src_normals->points[i].x = source_cloud->points[i].x;
      src_normals->points[i].y = source_cloud->points[i].y;
      src_normals->points[i].z = source_cloud->points[i].z;
  }

  //Cloud normals target
  cout << "Computing target cloud normals" << endl;
  pcl::PointCloud<pcl::PointNormal>::Ptr tar_normals (new pcl::PointCloud<pcl::PointNormal>); //Guardar normales de target
  ne.setInputCloud(target_cloud);
  ne.setSearchMethod(tree_xyz);
  ne.setRadiusSearch(0.05);
  ne.compute(*tar_normals);
  for(size_t i = 0;  i < tar_normals->points.size(); ++i) {
      tar_normals->points[i].x = target_cloud->points[i].x;
      tar_normals->points[i].y = target_cloud->points[i].y;
      tar_normals->points[i].z = target_cloud->points[i].z;
  }

  cout << "---------------------------" << endl;

  //SIFT source with normals
  pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale>::Ptr src_keypoints (new pcl::PointCloud<pcl::PointWithScale>);
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointNormal> ());
  sift.setSearchMethod(tree_normal);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(src_normals);
  sift.compute(*src_keypoints);

  cout << "Found " << src_keypoints->points.size () << " SIFT keypoints in source cloud" << endl;

  //SIFT target with normals
  pcl::PointCloud<pcl::PointWithScale>::Ptr tar_keypoints (new pcl::PointCloud<pcl::PointWithScale>);
  sift.setSearchMethod(tree_normal);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(tar_normals);
  sift.compute(*tar_keypoints);

  cout << "Found " << tar_keypoints->points.size () << " SIFT keypoints in target cloud" << endl;

  //Extract Features from SIFT Keypoints
  //Source
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_keypoints_xyz (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud (*src_keypoints, *src_keypoints_xyz);
  pcl::FPFHEstimation<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh; //Instancia fpfh
  fpfh.setSearchSurface (source_cloud);
  fpfh.setInputCloud (src_keypoints_xyz);
  fpfh.setInputNormals (src_normals);
  fpfh.setSearchMethod (tree_xyz);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features (new pcl::PointCloud<pcl::FPFHSignature33>());
  fpfh.setRadiusSearch(0.05);
  fpfh.compute(*src_features);

  cout << "Computed " << src_features->size() << " FPFH features for source cloud" << endl;

  //histogram_calculation

  //Target
  pcl::PointCloud<pcl::PointXYZ>::Ptr tar_keypoints_xyz (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::copyPointCloud (*tar_keypoints, *tar_keypoints_xyz);
  fpfh.setSearchSurface (target_cloud);
  fpfh.setInputCloud (tar_keypoints_xyz);
  fpfh.setInputNormals (tar_normals);
  fpfh.setSearchMethod (tree_xyz);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr tar_features (new pcl::PointCloud<pcl::FPFHSignature33>());
  fpfh.setRadiusSearch(0.05);
  fpfh.compute(*tar_features);

  cout << "Computed " << tar_features->size() << " FPFH features for target cloud" << endl;

  cout << "---------------------------" << endl;

//Utilizamos SAC-IA para conseguir el "rough alignment". Viene a ser un findCorrespondences entre features

  Eigen::Matrix4f max_tr = Eigen::Matrix4f::Identity(); //Definimos la matriz
  max_tr = computeInitialAlignment (src_keypoints, src_features, tar_keypoints, tar_features);

  cout << "---------------------------" << endl;
  Eigen::Matrix4f max_icp = Eigen::Matrix4f::Identity(); //Definimos la matriz
  max_icp = refine_icp (source_cloud, target_cloud, max_tr);


  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud (*source_cloud, *transformed_cloud, max_icp);
  cout << "Calculated transform, saved in PCD " << endl;


  pcl::visualization::PCLVisualizer viewer ("PCL Viewer");

  int v1 (0);
  int v2 (1);
  viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
  viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);
  // Show keypoints in 3D viewer
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color_handler (source_cloud, 0, 0, 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithScale> src_keypoints_color_handler (src_keypoints, 0, 255, 0);
  viewer.addPointCloud<pcl::PointXYZ> (source_cloud, source_color_handler, "source cloud",v1);
  viewer.addPointCloud<pcl::PointWithScale> (src_keypoints, src_keypoints_color_handler, "source keypoints",v1);
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "source keypoints",v1);

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transform_color_handler (transformed_cloud, 255, 255, 255);
  viewer.addPointCloud<pcl::PointXYZ> (transformed_cloud, transform_color_handler, "transform cloud",v2);


  while(!viewer.wasStopped ())
  {
  viewer.spinOnce ();
  }

  ros::spinOnce();

  return 0;
}

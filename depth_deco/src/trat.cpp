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
#include <pcl/correspondence.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <depth_deco/ROI_identifier.h>
#include "DEOpt/DEOpt.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl/registration/correspondence_types.h>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/correspondence_estimation.h>
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
#include <pcl/registration/icp_nl.h>
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
#include <pcl/surface/mls.h>

using namespace cv;
using namespace std;
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

ros::Publisher pub;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

PointCloud::Ptr show (new PointCloud);
PointCloud::Ptr show_prev (new PointCloud);

std::vector<pcl::PointCloud<pcl::PointXYZRGB>, Eigen::aligned_allocator<pcl::PointXYZRGB > > outputs;
Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();


Eigen::Matrix4f pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, int i){


  cout << "" << endl;
  cout << "ICP process: " << endl;

  PointCloud::Ptr src (new PointCloud);
  PointCloud::Ptr tgt (new PointCloud);


  vector<int> nan_idx;
  pcl::removeNaNFromPointCloud (*cloud_src,*cloud_src,nan_idx);
  pcl::removeNaNFromPointCloud (*cloud_tgt,*cloud_tgt,nan_idx);

  pcl::CorrespondencesPtr correspondences (new pcl::Correspondences);

  pcl::registration::CorrespondenceEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB> est;
  est.setInputSource (cloud_src);
  est.setInputTarget (cloud_tgt);

  est.determineCorrespondences (*correspondences,0.05f);



  cout << "Size source: " << cloud_src->size() << endl;
  cout << "Size target: " << cloud_tgt->size() << endl;
  cout << "Size correspondence: " << correspondences->size() << endl;

  cout << "Correspondences score: " << (correspondences->size()*100)/cloud_src->size() << endl;

  /*std::stringstream ss;

  if ((correspondences->size()*100)/cloud_src->size() < 90){

    ss << "/home/diego/filtro3/" << i << ".pcd";
    pcl::io::savePCDFile<pcl::PointXYZRGB>(ss.str(), *cloud_src);
    cout << "Nube " << i << "a " << i+1 << "no pasa el filtro " << endl;
    ss.str(std::string());
  }*/

  pcl::VoxelGrid<PointT> grid;
  grid.setLeafSize (0.01, 0.01, 0.01);
  grid.setInputCloud (cloud_src);
  grid.filter (*src);
  grid.setInputCloud (cloud_tgt);
  grid.filter (*tgt);

  // Compute surface normals and curvature
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);

  pcl::NormalEstimation<PointT, PointNormalT> norm_est;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
  norm_est.setSearchMethod (tree);
  norm_est.setKSearch (30);

  norm_est.setInputCloud (src);
  norm_est.compute (*points_with_normals_src);
  pcl::copyPointCloud (*src, *points_with_normals_src);

  norm_est.setInputCloud (tgt);
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  // Align
  pcl::IterativeClosestPointWithNormals<PointNormalT, PointNormalT> icp_nl;

  if ((correspondences->size()*100)/cloud_src->size() > 80){

    icp_nl.setTransformationEpsilon (1e-12);
    icp_nl.setEuclideanFitnessEpsilon(0.001);
    icp_nl.setMaxCorrespondenceDistance (0.05f);
    icp_nl.setMaximumIterations (500);
    icp_nl.setRANSACOutlierRejectionThreshold (0.0005);
  }
  else{
    icp_nl.setTransformationEpsilon (1e-18);
    icp_nl.setEuclideanFitnessEpsilon(0.000005);
    icp_nl.setMaxCorrespondenceDistance (0.05f);
    icp_nl.setMaximumIterations (1000000);
    icp_nl.setRANSACOutlierRejectionThreshold (0.05);
  }



  icp_nl.setInputSource (points_with_normals_src);
  icp_nl.setInputTarget (points_with_normals_tgt);

	Eigen::Matrix4f targetToSource;
	PointCloudWithNormals::Ptr reg_result (new PointCloudWithNormals);

	icp_nl.align (*reg_result);

  cout << "ICP score: " << icp_nl.getFitnessScore() << endl;
	cout << "Transformation Matrix: " << endl;
  cout << icp_nl.getFinalTransformation() << endl;
  cout << endl;


	targetToSource = icp_nl.getFinalTransformation();


  return targetToSource;

}

int main (int argc, char **argv){

  ros::init (argc, argv, "trat");

	ros::NodeHandle n;

  cout << endl;
  cout << "Cargando los archivos..." << endl;
  cout << "---------------------------" << endl;


  std::vector<pcl::PointCloud<pcl::PointXYZRGB>, Eigen::aligned_allocator<pcl::PointXYZ > > clouds;

  Eigen::Matrix4f individual_tr = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f global_tr = Eigen::Matrix4f::Identity();

	std::stringstream ss;

  for (int i = 1 ; i < 279; i+=2){

    ss << "/home/carba/Downloads/horizontal/" << i << ".pcd";
    //Temp point clound contrainer to load a point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZRGB>);
    //Load the point cloud
    pcl::io::loadPCDFile<pcl::PointXYZRGB>(ss.str(), *cloud_temp);
    //Once point cloud is loaded push back the point cloud into the vector
    clouds.push_back(*cloud_temp);

    ss.str(std::string());
  }

  cout << clouds.size() << " clouds loaded " << endl;

  pub = n.advertise<sensor_msgs::PointCloud2> ("cloud_tr",100);

	PointCloud::Ptr source (new PointCloud), target (new PointCloud);
  PointCloud::Ptr output (new PointCloud);

	Eigen::Matrix4f pairTransform = Eigen::Matrix4f::Identity ();


  for (std::size_t i = 1; i < clouds.size (); ++i)
  {

    if (i==clouds.size()-1){
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_pcd (new pcl::PointCloud<pcl::PointXYZRGB>);
      copyPointCloud (*show, *result_pcd);
      sensor_msgs::PointCloud2 cloud_tr;
      pcl::io::savePCDFile<pcl::PointXYZRGB>("/home/carba/Downloads/horizontal/result.pcd", *result_pcd);
      pcl::toROSMsg(*show,cloud_tr);
      std_msgs::Header h = cloud_tr.header;
      cloud_tr.header.frame_id = "world";
      pub.publish (cloud_tr);
      break;
    }
    else{
      copyPointCloud (clouds[i], *target);
      copyPointCloud (clouds[i+1], *source);
    }


    cout << "Transformada " << i << endl;

    // Add visualization data

    individual_tr = pairAlign (source, target, i);

    GlobalTransform *= individual_tr; //Actualizamos GlobalTransform

  	pcl::transformPointCloud (*source, *output, GlobalTransform);

  	//*output += *cloud_tgt;

    *show_prev += *output;

    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize (0.02, 0.02, 0.02);
    grid.setInputCloud (show_prev);
    grid.filter (*show_prev);

    *show += *show_prev;

    sensor_msgs::PointCloud2 cloud_tr;

    pcl::toROSMsg(*show,cloud_tr);
    std_msgs::Header h = cloud_tr.header;
    cloud_tr.header.frame_id = "world";
    pub.publish (cloud_tr);

    //outputs.push_back (*show);



    /*pcl::visualization::PCLVisualizer viewer ("PCL Viewer");

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> show_h (result);
    viewer.addPointCloud<pcl::PointXYZRGB> (result, show_h, "output");


    viewer.spinOnce ();
  */

    cout << "Nube final tiene " << show->size() << " puntos" << endl;



    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_pcd (new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud (*show, *result_pcd);
    pcl::io::savePCDFile<pcl::PointXYZRGB>("/home/carba/Downloads/horizontal/result.pcd", *result_pcd);

    cout << endl;
    cout << "Global Transform: " << endl;
    cout << GlobalTransform << endl;


  }


	cout << "Fin " << endl;


	ros::spin();

  return 0;
}

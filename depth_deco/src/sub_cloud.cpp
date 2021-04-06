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

ros::Publisher pub;

pcl::PointCloud<pcl::Normal>::Ptr normals_ (new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::FPFHSignature33>::Ptr features_ (new pcl::PointCloud<pcl::FPFHSignature33>);
pcl::search::KdTree<pcl::PointXYZ>::Ptr search_method (new pcl::search::KdTree<pcl::PointXYZ>);

//Target es la nube la cual queremos que se alineen las demás
//Añadimos templates para alinear al target

std::vector<sensor_msgs::PointCloud2ConstPtr> template_list; //Lista de nubes crudas
std::vector<Eigen::Matrix4f> matrix_tr; //Vector de matrices con transformaciones
pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);

bool first_in=false;
int i=0;

/*void Template_Alignment (const pcl::PointCloud<pcl::Normal>::Ptr normals_, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features){

	pcl::SampleConsensusInitialAlignment <pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	pcl::PointCloud<pcl::PointXYZ>::Ptr template_cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ> registration_output;

	//Initialize parameters

	cout << "Realizando SAC-IA"<<endl;

	sac_ia.setMinSampleDistance (0.05);
	sac_ia.setMaxCorrespondenceDistance (0.01*0.01);
	sac_ia.setMaximumIterations (500);

	sac_ia.setInputTarget (target_cloud);
	sac_ia.setTargetFeatures (features_);

//Template
	pcl::fromROSMsg (*template_list[i],*template_cloud);

	vector<int> mapping1;
	removeNaNFromPointCloud (*template_cloud, *template_cloud, mapping1);

	sac_ia.setInputSource (template_cloud);
	sac_ia.setSourceFeatures (features_);
	sac_ia.align (registration_output);

	cout << "Fitness Score: " << sac_ia.getFitnessScore() << endl;
	cout << "Final Transformation: " << sac_ia.getFinalTransformation () << endl;


	i++;
}

void processInput (const pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud){

	vector<int> mapping1;
	removeNaNFromPointCloud (*target_cloud, *target_cloud, mapping1);


	pcl::NormalEstimation <pcl::PointXYZ, pcl::Normal> norm_est;
	norm_est.setInputCloud (target_cloud);
	norm_est.setSearchMethod (search_method);
	norm_est.setRadiusSearch (0.02f); //MODIFICAR
	norm_est.compute (*normals_);

	pcl::FPFHEstimation <pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
	fpfh_est.setInputCloud (target_cloud);
	fpfh_est.setInputNormals (normals_);
	fpfh_est.setSearchMethod (search_method);
	fpfh_est.setRadiusSearch (0.02f); //MODIFICAR
	fpfh_est.compute (*features_);
}*/



/*void list_cloud (const sensor_msgs::PointCloud2ConstPtr cloud_msg){
	//Recibimos de la kinect
	std::vector<pcl::PointCloud<pcl::PointXYZ>, Eigen::aligned_allocator<pcl::PointXYZ> > clouds;

	std::stringstream ss;

        for (int i = 0; i<10; i++)
        {
                ss << "file" << i << ".pcd";
                //Temp point clound contrainer to load a point cloud
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);

                //Load the point cloud
                pcl::io::loadPCDFile<pcl::PointXYZ>(ss.str(), *cloud_temp);

                //Once point cloud is loaded push back the point cloud into the vector
                clouds.push_back(*cloud_temp);

                ss.str(std::string());
				}
}*/


int main (int argc, char **argv){

  ros::init (argc, argv, "sub_cloud");


	ros::NodeHandle n;

	//ros::Subscriber sub = n.subscribe ("cloud_publish",1,list_cloud);


	//pub = n.advertise<sensor_msgs::PointCloud2> ("cloud_with_icp",100);

	/*std::vector<pcl::PointCloud<pcl::PointXYZRGB>, Eigen::aligned_allocator<pcl::PointXYZRGB > > clouds;

	std::stringstream ss;

        for (int i = 1; i<100; ++i)
        {
                ss << "/home/diego/horizontal/" << i << ".pcd";
                //Temp point clound contrainer to load a point cloud
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZRGB>);

                //Load the point cloud
                pcl::io::loadPCDFile<pcl::PointXYZRGB>(ss.str(), *cloud_temp);

                //Once point cloud is loaded push back the point cloud into the vector
                clouds.push_back(*cloud_temp);

                ss.str(std::string());
				}

	cout << "Size: " << clouds.size() << endl;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_5 (new pcl::PointCloud<pcl::PointXYZRGB>);
	copyPointCloud (clouds[1], *cloud_5);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_6 (new pcl::PointCloud<pcl::PointXYZRGB>);
	copyPointCloud (clouds[77], *cloud_6);



  pcl::PointCloud<pcl::PointXYZRGB>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGB>);

  *result = *cloud_5;
  *result += *cloud_6;*/

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr result (new pcl::PointCloud<pcl::PointXYZRGB>);

  pcl::io::loadPCDFile ("/home/diego/ICP_PCL/result.pcd",*result );



		pcl::visualization::PCLVisualizer viewer ("PCL Viewer");

	  /*int v1 (0);
	  int v2 (1);
	  viewer.createViewPort (0.0, 0.0, 0.5, 1.0, v1);
	  viewer.createViewPort (0.5, 0.0, 1.0, 1.0, v2);

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_5_h (cloud_5, 255, 255, 255);
		viewer.addPointCloud<pcl::PointXYZ> (cloud_5, cloud_5_h, "cloud 5",v1);*/
		pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> cloud_6_h (result);
		viewer.addPointCloud<pcl::PointXYZRGB> (result, cloud_6_h, "result");



	  while(!viewer.wasStopped ())
	  {
			viewer.spin ();

	}

	ros::spin();

  return 0;
}

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <limits>
#include <vector>
#include <Eigen/Core>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/opencv.hpp"

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


std::vector<sensor_msgs::PointCloud2ConstPtr> cloudlist;
std::vector<sensor_msgs::ImageConstPtr> imglist;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
int j=1;
ros::Publisher pub;

std::vector<double> scaleDepth(Mat img, float range) // parameter obtention to scale a depth image(greyscale) between a distance range referring to the average to focus on

{	//Average, min, max pixel calculation on ROI
        float sum=0;
        float avg=0;
        int pixel_info=0;
        double min, max;
        Mat mask = img>0 & img<4000;

        cv::minMaxLoc(img, &min, &max, NULL, NULL, mask);

        for(int j=0;j<img.rows;j++)
           {
            for (int i=0;i<img.cols;i++)
            {   std::cout << "Valor pixel:" << img.at<float>(j,i) << std::endl;
		if (img.at<float>(j,i)>0)
                {pixel_info=pixel_info +1;}

             sum= sum + img.at<float>(j,i);
            }
           }
        avg= sum / pixel_info;

//        if ((max-avg)>=(avg-min))
//            range=max-avg;
//        else
//            range=avg-min;

        // Converting ROI to 8bit and scaling distances between 0 and 255
        double alpha=255/(2*range);
        double beta=255/((avg+range)/(range-avg)+1);

        std::vector<double> conVparams(5);
        conVparams[0]=alpha;
        conVparams[1]=beta;
        conVparams[2]=avg;
        conVparams[3]=min;
        conVparams[4]=max;
        return conVparams;
}

void processing (const sensor_msgs::Image msg, const PointCloud::Ptr cloud,int i,sensor_msgs::PointCloud2ConstPtr cloud_msg)// 2D/3D processing to detect blast holes (main)
{

				cv_bridge::CvImagePtr cv_ptr;
				cv::Mat current_frame;
				Mat idepth, idepth_scaled;
				std::vector<double> scale_params(5);


			  try
			  {


					idepth=cv_bridge::toCvCopy(msg, "32FC1")->image; //conversion
					float range=200;
  					std::cout << "Depth Image Scaling Info"<< std::endl << std::endl;
 					scale_params=scaleDepth(idepth,range);  //scale param obtention to a distance range
 					std::cout << "Average:" << scale_params[2] << std::endl;
  					std::cout << "Min:" << scale_params[3] << std::endl;
 					std::cout << "Max:" << scale_params[4] << std::endl<<endl;
  					std::cout << "__________________________"<< endl;
  					idepth.convertTo(idepth_scaled, CV_8UC1,scale_params[0], scale_params[1]);//scaling

				}
				catch (cv_bridge::Exception& e)
				{
					ROS_ERROR("cv_bridge exception: %s", e.what());
					return;
				}


         std::cout << "Cloud seq:" << cloud_msg->header.seq  << endl;
         std::cout << "Depth seq:" << msg.header.seq  << endl;

				 cout << "Image list: " << imglist.size() << endl;
				 cout << "Cloud list: " << cloudlist.size() << endl;

				 std::stringstream ss;


				 ss << "/home/carba/Downloads/Archivos scanmatching mina/pruebanube/" << j << ".pcd";
		                 pcl::io::savePCDFile<pcl::PointXYZRGB>(ss.str(), *cloud);

         std::stringstream ss1;


         ss1 << "/home/carba/Downloads/Archivos scanmatching mina/pruebaimg/" << j << ".jpg";

 					imwrite( ss1.str(), idepth_scaled);
 					ss1.str(std::string());


				 ss.str(std::string());
	j=j+1;	


}

void cloud_cb (sensor_msgs::PointCloud2ConstPtr cloud_msg)// point cloud callback: kinect point cloud buffer
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

	cloudlist.push_back(cloud_msg);

	pcl::fromROSMsg (*cloud_msg, *cloud);
   sensor_msgs::Image im_msg;
   for (int i=0;i<imglist.size();i++){
        if (imglist[i]->header.seq==cloud_msg->header.seq){
           im_msg=*imglist[i];
           //cloudlist.clear();
           //imglist.clear();
           processing(im_msg,cloud,i, cloud_msg);
        }

   }
}




void depth_cb (const sensor_msgs::ImageConstPtr im_msg)//depth image callback: kinect depth image buffer
{

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    sensor_msgs::PointCloud2ConstPtr cloud_msg;
   imglist.push_back(im_msg);
   for (int i=0;i<cloudlist.size();i++){
       if (cloudlist[i]->header.seq==im_msg->header.seq){
           cloud_msg=cloudlist[i];
					 pcl::fromROSMsg (*cloud_msg, *cloud);
           //cloudlist.clear();
           //imglist.clear();
           processing(*im_msg,cloud,i,cloud_msg);
        }
   }
}


int main (int argc, char **argv){

	ros::init(argc, argv, "get_cloud");
	ros::NodeHandle n;




	ros::Subscriber sub = n.subscribe ("/kinect2/sd/image_depth", 1, depth_cb);
	// Create a ROS subscriber for the input point cloud
	ros::Subscriber sub2 = n.subscribe ("/kinect2/hd/points", 1, cloud_cb);
	ros::spin();


	return 0;
}

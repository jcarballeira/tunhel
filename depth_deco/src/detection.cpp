/*
  * transformarlos con la global transformation
 * comparar,descartar y almacenar
 * mostrarlos
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * */#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>
#include<ctime>
#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>

//OpenCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

//ROS
#include <ros/ros.h>
#include <sensor_msgs/RegionOfInterest.h>
#include <depth_deco/ROI_identifier.h>
#include "DEOpt/DEOpt.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

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


typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
std::vector<sensor_msgs::PointCloud2ConstPtr> cloudlist;// buffer temp nubes para cuadrar por header.seq
std::vector<sensor_msgs::ImageConstPtr> imglist;// buffer temp imagenes para cuadrar por header.seq

std::vector<Mat> depthlist;//vector imagenes profundidad
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr, Eigen::aligned_allocator<pcl::PointXYZ > > clouds;//vector nubes
std::vector<Point3d> acc_centres;


PointCloud::Ptr show (new PointCloud);
PointCloud::Ptr show_prev (new PointCloud);

Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity ();
int x=0;
int nubes_a_analizar;
bool hole_detection=true;
bool directo=true;
std::string frame;
ofstream centersFile;

ros::Publisher pub;
ros::Publisher pub_ROIlocation;
ros::Publisher pub_ROIcloud;
ros::Publisher pub_center;
ros::Publisher pub_HoleCloud;
ros::Publisher pub_ScanmCloud;
ros::Publisher pub_Cloud;
ros::Publisher pub_CylPoints;
ros::Publisher pub_Arrow;
ros::Publisher pub_Arrows;

struct CV_EXPORTS Center
{   Point2d location;
      double radius;
      double confidence;
  };


struct parameters {//blob detection parameter structure
    float thresholdStep;
    float minThreshold;
    float maxThreshold;
    size_t minRepeatability;
    float minDistBetweenBlobs;

    bool filterByColor;
    unsigned char blobColor;

    bool filterByArea;
    float minArea;
    float maxArea;

    bool filterByCircularity;
    float minCircularity;
    float maxCircularity;

    bool filterByInertia;
    //minInertiaRatio = 0.6;
    float minInertiaRatio;
    float maxInertiaRatio;

    bool filterByConvexity;
    //minConvexity = 0.8;
    float minConvexity;
    float maxConvexity;
} params;

Eigen::Matrix4f pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt){


  cout << "" << endl;
  cout << "    ICP process: " << endl;

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



  cout << "    -Size source: " << cloud_src->size() << endl;
  cout << "    -Size target: " << cloud_tgt->size() << endl;
  cout << "    -Size correspondence: " << correspondences->size() << endl;

  cout << "    -Correspondences score: " << (correspondences->size()*100)/cloud_src->size() << endl;



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

  cout << "    -ICP score: " << icp_nl.getFitnessScore() << endl;
  cout << "    -Indiv. Transformation Matrix: " << endl;
  cout << icp_nl.getFinalTransformation() << endl;
  cout << endl;


	targetToSource = icp_nl.getFinalTransformation();


  return targetToSource;

}

Point3d cloudSubset(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, Mat im_scan, Mat im_msg, depth_deco::ROI_identifier ROI_scan, depth_deco::ROI_identifier ROI_msg, int hole_id) //Point cloud hole segmentation and publishing (hole and surroundings)
{

    std::cerr << "Dentro Inside" << endl;
    bool fp=false;
    Point3d centre;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ROI (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_hole (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rest (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_scanm (new pcl::PointCloud<pcl::PointXYZRGB>);

    //pcl::fromROSMsg (*cld_msg, *cloud);

    pcl::ExtractIndices<pcl::PointXYZRGB> extract_zoom;
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_hole;
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_rest;
     pcl::ExtractIndices<pcl::PointXYZRGB> extract_scanm;

    pcl::PointIndices::Ptr inliers_zoom (new pcl::PointIndices ());
    pcl::PointIndices::Ptr inliers_hole (new pcl::PointIndices ());
    pcl::PointIndices::Ptr inliers_rest (new pcl::PointIndices ());
    pcl::PointIndices::Ptr inliers_scanm (new pcl::PointIndices ());

    Point center(ROI_scan.width/2,ROI_scan.height/2);

    double radius = ROI_scan.width/28;

    for (int i=0; i< im_scan.cols; i++){
           for (int j=0; j<im_scan.rows; j++){
             double distancia=sqrt(pow((i-center.x),2)+pow((j-center.y),2));

             if (distancia>radius*2){
                inliers_scanm->indices.push_back( (j+ROI_scan.y_offset)*512 + (i+ROI_scan.x_offset) );
             }
           }
       }


       for (int i=0; i< im_msg.cols; i++){
        for (int j=0; j<im_msg.rows; j++){
	  	
          inliers_zoom->indices.push_back( (j+ROI_msg.y_offset)*512 + (i+ROI_msg.x_offset) );
          Scalar intensity = im_msg.at<uchar>(i, j);
          //std::cerr << "Prueba img " << intensity.val[0] << std::endl;
          if (intensity.val[0]==255){
             inliers_hole->indices.push_back( (j+ROI_msg.y_offset)*512 + (i+ROI_msg.x_offset) );
          }
          else{
              inliers_rest->indices.push_back( (j+ROI_msg.y_offset)*512 + (i+ROI_msg.x_offset) );
          }
        }
    }


    extract_zoom.setInputCloud (cloud);
    extract_zoom.setIndices (inliers_zoom);
    extract_zoom.setNegative (false);
    extract_zoom.filter (*cloud_ROI);

    extract_hole.setInputCloud (cloud);
    extract_hole.setIndices (inliers_hole);
    extract_hole.setNegative (false);
    extract_hole.filter (*cloud_hole);

    extract_rest.setInputCloud (cloud);
    extract_rest.setIndices (inliers_rest);
    extract_rest.setNegative (false);
    extract_rest.filter (*cloud_rest);

    extract_scanm.setInputCloud (cloud);
    extract_scanm.setIndices (inliers_scanm);
    extract_scanm.setNegative (false);
    extract_scanm.filter (*cloud_scanm);

    /*pcl::PointCloud<pcl::Normal>::Ptr norm_cloud;
    norm_cloud=calc_normals(cloud_hole);
    normalsVis(cloud_hole,norm_cloud);*/

    sensor_msgs::PointCloud2 cloud_ROI_msg;
    pcl::toROSMsg (*cloud_ROI, cloud_ROI_msg);
    cloud_ROI_msg.header.frame_id = frame;
    pub_ROIcloud.publish (cloud_ROI_msg);

    sensor_msgs::PointCloud2 cloud_Hole_msg;
    pcl::toROSMsg (*cloud_hole, cloud_Hole_msg);
    cloud_Hole_msg.header.frame_id = frame;
    pub_HoleCloud.publish (cloud_Hole_msg);

    sensor_msgs::PointCloud2 cloud_Scanm_msg;
    pcl::toROSMsg (*cloud_scanm, cloud_Scanm_msg);
    cloud_Scanm_msg.header.frame_id = frame;
    pub_ScanmCloud.publish (cloud_Scanm_msg);

    //cylinderModel_normals(cloud_ROI);
    //CylMarker();

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg (*cloud, cloud_msg);	
    cloud_ROI_msg.header.frame_id = frame;
    pub_Cloud.publish (cloud_msg);

    double z;
    double minz=1000000;
    double maxz=0;	
    for (int i=0;i<cloud_ROI->points.size();i++){

        z=cloud_ROI->points[i].z;
        if ((!isnan(z))&&(minz>z)){minz=z;}
	else if ((!isnan(z))&&(maxz<z)){maxz=z;}
    }

    double center_x=cloud_ROI->points[ROI_msg.height*ROI_msg.width/2].x;
    double center_y=cloud_ROI->points[ROI_msg.height*ROI_msg.width/2].y;
    double center_z=minz;

    if ((!isnan(center_z))&&(!isnan(center_x))&&(!isnan(center_y))){

        centre.x=center_x;
        centre.y=center_y;
        centre.z=center_z;
        double distance=sqrt(center_x*center_x+center_y*center_y+center_z*center_z);

        //std::cerr << "Location (x,y,z): [" << center_x <<","<< center_y <<","<< center_z <<"]"<< std::endl;
        //std::cerr << "Distance:" << distance <<"meters"<< std::endl;

    }



    /// parte para guardar las distintas nubes en formato .pcd
    if ((!isnan(cloud_ROI->points[ROI_msg.height*ROI_msg.width/2].x))&&(!isnan(cloud_ROI->points[ROI_msg.height*ROI_msg.width/2].y))&&(!isnan(minz))){
	
        centersFile << center_x << "," << center_y << "," << center_z << endl;
        //Writing whole cloud .pcd
        std::ostringstream os;
        os << "cloud" << hole_id+1 << ".pcd";
        string va=os.str();
        const char *filenam=va.c_str();

        pcl::io::savePCDFileASCII (filenam, *cloud);

        //Writing rest cloud .pcd
        std::ostringstream osv;
        osv << "rest" << hole_id+1 << ".pcd";
        string varv=osv.str();
        const char *filenamev=varv.c_str();

        pcl::io::savePCDFileASCII (filenamev, *cloud_rest);

        //Writing hole cloud .pcd
        std::ostringstream oss;
        oss << "hole" << hole_id+1 << ".pcd";
        string var=oss.str();
        const char *filename=var.c_str();

        pcl::io::savePCDFileASCII (filename, *cloud_hole);

        //std::cerr << "Saved " << cloud_hole->points.size () << " data  points to hole.pcd." << std::endl;
        std::cerr << "Hole points: " << cloud_hole->points.size() << endl;

        /*//Filtering ROI cloud to extract entrance wall rests
	pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud (*cloud_ROI);
    	pass.setFilterFieldName ("z");
    	pass.setFilterLimits (minz+0.01, maxz-0.015);
    	//pass.setFilterLimitsNegative (true);
    	pass.filter (*cloud_ROI);
        */
	//Writing ROI cloud .pcd
        std::ostringstream ost;
        ost << "roi" << hole_id+1 << ".pcd";
        string vart=ost.str();
        const char *filenamet=vart.c_str();

        pcl::io::savePCDFileASCII (filenamet, *cloud_ROI);


        //Writing sacn matching cloud .pcd
        std::ostringstream osz;
        osz << "scanm" << hole_id+1 << ".pcd";
        string varz=osz.str();
        const char *filenamez=varz.c_str();

        pcl::io::savePCDFileASCII (filenamez, *cloud_scanm);

        
    }
    else {

	std::cerr << "Hole discarded " << endl;
        fp=true; }

    //std::cout << "__________________________"<<endl<<endl;

    return centre;
}

Mat CircleDetection(Mat res_image) //Hough circle detection & mask the interior of the detected circle

{
        vector<Vec3f> circles;
        Mat image=res_image.clone();

        erode(image, image, 0, Point(-1,-1), 2, 1, 1 );
        dilate(image, image, 0, Point(-1,-1), 2, 1, 1 );
        Canny( image, image, 5, 70, 3 );
        //namedWindow("Canny", CV_WINDOW_AUTOSIZE);
        //imshow("Canny", image);
        moveWindow("Canny", 512+375,360);
        HoughCircles(image, circles, CV_HOUGH_GRADIENT,1,20, 180, 10,0,200);
        //std::cout << "Found circles:" <<circles.size() << std::endl;

        Mat maskedImage=res_image;
        Mat mask(image.size(),image.type());
        mask.setTo(cv::Scalar(0));

        // Draw circles
        for( size_t i = 0; i < circles.size(); i++ )
           {
           Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));

           int radius = cvRound(circles[i][2]);

           // draw the circle outline
           circle( mask, center, radius, Scalar(255), -1, 8, 0 );
           res_image.copyTo(maskedImage,mask);
           ;

        }
        return maskedImage;
}

cv::Rect ROI_w_limit_comprobation (Mat img, float size, KeyPoint kypt) //ROI selection over holes considering the borders not to step out the image

{
    int cols=img.cols;
    int rows=img.rows;
    Rect rectangle;
    float x_ini=kypt.pt.x-size/2;
    float y_ini=kypt.pt.y-size/2;
    float x_size=size;
    float y_size=size;

    if (kypt.pt.x-size/2 < 0)
        {x_ini=0;
         x_size=size/2+kypt.pt.x;}

    if (kypt.pt.y-size/2 < 0)
        {y_ini=0;
         y_size=size/2+kypt.pt.y;}

    if (kypt.pt.x+size/2 > cols)
        {x_ini=kypt.pt.x-size/2;
         x_size=size/2+cols-kypt.pt.x;}

    if (kypt.pt.y+size/2 > rows)
        {y_ini=kypt.pt.y-size/2;
         y_size=size/2+rows-kypt.pt.y;}

    rectangle=Rect(x_ini,y_ini,x_size,y_size);

    return rectangle;
}

void findBlobs(InputArray _image, InputArray _binaryImage, std::vector<Center> &centers)
{
    //CV_INSTRUMENT_REGION()

    Mat image = _image.getMat(), binaryImage = _binaryImage.getMat();
    (void)image;
    centers.clear();

    std::vector < std::vector<Point> > contours;
    Mat tmpBinaryImage = binaryImage.clone();
    findContours(tmpBinaryImage, contours, RETR_LIST, CHAIN_APPROX_NONE);
//    namedWindow("canny", CV_WINDOW_AUTOSIZE);
//    imshow("canny", tmpBinaryImage);


    for (size_t contourIdx = 0; contourIdx < contours.size(); contourIdx++)
    {
        Center center;
        center.confidence = 1;
        Moments moms = moments(Mat(contours[contourIdx]));

        if (params.filterByArea)
        {
            double area = moms.m00;
            if (area < params.minArea || area >= params.maxArea)
                continue;
        }

        if (params.filterByCircularity)
        {
            double area = moms.m00;
            double perimeter = arcLength(Mat(contours[contourIdx]), true);
            double ratio = 4 * CV_PI * area / (perimeter * perimeter);
            if (ratio < params.minCircularity || ratio >= params.maxCircularity)
                continue;
        }

        if (params.filterByInertia)
        {
            double denominator = std::sqrt(std::pow(2 * moms.mu11, 2) + std::pow(moms.mu20 - moms.mu02, 2));
            const double eps = 1e-2;
            double ratio;
            if (denominator > eps)
            {
                double cosmin = (moms.mu20 - moms.mu02) / denominator;
                double sinmin = 2 * moms.mu11 / denominator;
                double cosmax = -cosmin;
                double sinmax = -sinmin;

                double imin = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmin - moms.mu11 * sinmin;
                double imax = 0.5 * (moms.mu20 + moms.mu02) - 0.5 * (moms.mu20 - moms.mu02) * cosmax - moms.mu11 * sinmax;
                ratio = imin / imax;
            }
            else
            {
                ratio = 1;
            }

            if (ratio < params.minInertiaRatio || ratio >= params.maxInertiaRatio)
                continue;

            center.confidence = ratio * ratio;
        }

        if (params.filterByConvexity)
        {
            std::vector < Point > hull;
            convexHull(Mat(contours[contourIdx]), hull);
            double area = contourArea(Mat(contours[contourIdx]));
            double hullArea = contourArea(Mat(hull));
            double ratio = area / hullArea;
            if (ratio < params.minConvexity || ratio >= params.maxConvexity)
                continue;
        }

        if(moms.m00 == 0.0)
            continue;
        center.location = Point2d(moms.m10 / moms.m00, moms.m01 / moms.m00);

        if (params.filterByColor)
        {
            if (binaryImage.at<uchar> (cvRound(center.location.y), cvRound(center.location.x)) != params.blobColor)
                continue;
        }

        //compute blob radius
        {
            std::vector<double> dists;
            for (size_t pointIdx = 0; pointIdx < contours[contourIdx].size(); pointIdx++)
            {
                Point2d pt = contours[contourIdx][pointIdx];
                dists.push_back(norm(center.location - pt));
            }
            std::sort(dists.begin(), dists.end());
            center.radius = (dists[(dists.size() - 1) / 2] + dists[dists.size() / 2]) / 2.;
        }

        centers.push_back(center);
  // std::cout << "Contour center " << center.location << endl;
    // std::cout << "Area=" <<moms.m00<<  std::endl;
      //  std::cout << "_______________________________________________"<<  std::endl;

    }
}

std::vector<KeyPoint> BlobDetection(Mat image, double dist){

  std::vector<KeyPoint> keypoints;
  std::vector < std::vector<Center> > centers;

  //Detection parameters
  // Binarization thresholds
  params.minThreshold = 0;
  params.maxThreshold = 255;
  params.thresholdStep=10;
  params.minRepeatability = 2;
  params.minDistBetweenBlobs = 10;

  //Filter by color
  params.filterByColor = true;
  params.blobColor = 255;

  // Filter by Area.
  params.filterByArea = true;
  //params.minArea = 20;
  //params.maxArea = 500;
  params.minArea = -7*dist/1200+24;
  params.maxArea = 195.4*(sin(dist/1200-3.1416))+18.34*(pow((dist/1000-10),2))-846;

  // Filter by Circularity
  params.filterByCircularity = false;
  params.minCircularity = 0.55;
  params.maxCircularity = std::numeric_limits<float>::max();

  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.85;
  params.maxConvexity = std::numeric_limits<float>::max();

  // Filter by Inertia
  params.filterByInertia = false;
  params.minInertiaRatio = 0.1;
  params.maxInertiaRatio = std::numeric_limits<float>::max();



  for (double thresh = params.minThreshold; thresh < params.maxThreshold; thresh += params.thresholdStep)
    {

    Mat binarizedImage;
    threshold(image, binarizedImage, thresh, 255, THRESH_BINARY);
    std::vector < Center > curCenters;
    std::vector < std::vector<Center> > newCenters;
    findBlobs(binarizedImage, binarizedImage, curCenters);

     for (size_t i = 0; i < curCenters.size(); i++)
          {
          bool isNew = true;
          for (size_t j = 0; j < centers.size(); j++)
              {
                  double dist = norm(centers[j][ centers[j].size() / 2 ].location - curCenters[i].location);
                  isNew = dist >= params.minDistBetweenBlobs && dist >= centers[j][ centers[j].size() / 2 ].radius && dist >= curCenters[i].radius;
                  if (!isNew)
                  {
                      centers[j].push_back(curCenters[i]);

                      size_t k = centers[j].size() - 1;
                      while( k > 0 && centers[j][k].radius < centers[j][k-1].radius )
                      {
                          centers[j][k] = centers[j][k-1];
                          k--;
                      }
                      centers[j][k] = curCenters[i];

                      break;
                  }
              }
              if (isNew) {
                  newCenters.push_back(std::vector<Center> (1, curCenters[i]));
              }
          }
      std::copy(newCenters.begin(), newCenters.end(), std::back_inserter(centers));

  }

  for (size_t i = 0; i < centers.size(); i++)
   {

          if (centers[i].size() < params.minRepeatability)
              continue;
          Point2d sumPoint(0, 0);
          double normalizer = 0;
          for (size_t j = 0; j < centers[i].size(); j++)
          {
              sumPoint += centers[i][j].confidence * centers[i][j].location;
              normalizer += centers[i][j].confidence;
          }
          sumPoint *= (1. / normalizer);

          KeyPoint kpt(sumPoint, (float)(centers[i][0].radius) * 2.0f);
          if((kpt.pt.x<440) && (kpt.pt.y<360)&&(kpt.pt.x>50) && (kpt.pt.y>50)){
          keypoints.push_back(kpt);
            }
   }

return keypoints;
}

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
            { //std::cout << "Valor pixel:" << img.at<float>(j,i) << std::endl;
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

void processing ()// 2D/3D processing to detect blast holes (main)
{

        //Check sizes
        cout << clouds.size() << " clouds loaded " << endl;
        cout << depthlist.size() << " depth images loaded " << endl;
        /*ros::Time begin = ros::Time::now();*/
        depth_deco::ROI_identifier ROI_msg;
        depth_deco::ROI_identifier ROI_scan;

        std::vector<Point3d>  centers_indcloud;

         //std::cout << "Cloud seq:" << cloud_msg->header.seq  << endl<< endl<< endl;
         //std::cout << "Depth seq:" << msg.header.seq  << endl<< endl<< endl;
         std::cout << "__________________________"<< endl;


         //Conversion of ros msg, scale image
         Mat idepth, idepth_scaled;
	     //Vectores de almacenamiento
  	 

	 
	 
for (int c=0; c<depthlist.size();c++){
         std::cout << "__________________________"<< endl;
         cout << "Nube:"<< c << endl<< endl;
         idepth=depthlist[c];
         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud=clouds[c];
         centers_indcloud.clear();
	
         //idepth=cv_bridge::toCvCopy(msg, "32FC1")->image; //conversion
         //namedWindow("Depth Or", CV_WINDOW_AUTOSIZE);
         //imshow("Depth Or", idepth );
         //imwrite( "../../images/Gray_Image.jpg", idepth );

         // Scaling to full depth resolution

         std::vector<double> scale_params(5);
         float range=200;
         if (directo==true){
         std::cout << "Depth Image Scaling Info:"<< std::endl << std::endl;
         scale_params=scaleDepth(idepth,range);  //scale param obtention to a distance range
         std::cout << "  -Average:" << scale_params[2] << std::endl;
         std::cout << "  -Min:" << scale_params[3] << std::endl;
         std::cout << "  -Max:" << scale_params[4] << std::endl<<endl;
         //std::cout << "__________________________"<< endl;
         idepth.convertTo(idepth_scaled, CV_8UC1,scale_params[0], scale_params[1]);//scaling
         }
         if (directo==false){
             idepth_scaled=idepth;
         }

         namedWindow("Depth", CV_WINDOW_AUTOSIZE); //idepth_scaled= scaled original image
         imshow("Depth", idepth_scaled);


        //Hole detection
        std::vector<KeyPoint> holes;
        if (hole_detection==true){

            holes=BlobDetection(idepth_scaled, scale_params[2]);
        }
        //std::cout << "Found holes:" <<holes.size() << std::endl;
        //std::cout << "__________________________"<< endl;

        Mat idepth_scaled_copy=idepth_scaled.clone();

       if (holes.size()!=0)  {
       for (int i=0; i<holes.size();i++){ 
        //std::cout << "Hole " << i+1 <<":" << holes[i].pt <<"(pixels)"<< endl;
        //ROIs (scan and hole isolating) in each hole, identifying them in pointcloud and identification of each hole's pixels
        int roi_width=holes[i].size*1.2;
        int roi_height=holes[i].size*1.2;
        Rect rectzoom =ROI_w_limit_comprobation(idepth_scaled, roi_width, holes[i]);
        //Rect rectzoom=Rect(holes[i].pt.x-roi_width/2,holes[i].pt.y-roi_height/2,roi_width,roi_height);
        ROI_msg.x_offset=(uint32_t)holes[i].pt.x-roi_width/2;
        ROI_msg.y_offset=(uint32_t)holes[i].pt.y-roi_height/2;
        ROI_msg.height=(uint32_t)roi_height;
        ROI_msg.width=(uint32_t)roi_width;
        pub_ROIlocation.publish (ROI_msg);

	int roi_scansize=holes[i].size*14;
        Rect scanzoom =ROI_w_limit_comprobation(idepth_scaled, roi_scansize, holes[i]);
        //Rect scanzoom=Rect(holes[i].pt.x-roi_scansize/2,holes[i].pt.y-roi_scansize/2,roi_scansize,roi_scansize);
        ROI_scan.x_offset=(uint32_t)holes[i].pt.x-roi_scansize/2;
        ROI_scan.y_offset=(uint32_t)holes[i].pt.y-roi_scansize/2;
        ROI_scan.height=(uint32_t)roi_scansize;
        ROI_scan.width=(uint32_t)roi_scansize;

        Mat zoom = idepth_scaled(rectzoom);
        //namedWindow("Zoom", CV_WINDOW_AUTOSIZE);
        //imshow("Zoom", zoom);

        Mat im_scan = idepth_scaled(scanzoom);
        //namedWindow("Scan Zoom", CV_WINDOW_AUTOSIZE);
        //imshow("Scan Zoom", im_scan);

        //equalize each hole image
        Mat zoomeq;
        equalizeHist(zoom,zoomeq);
        //histogram_calculation(zoom);
        //namedWindow("Equalized Zoom", CV_WINDOW_AUTOSIZE);
        //imshow("Equalized Zoom", zoomeq);


        //highlight each selected hole
        rectangle(idepth_scaled_copy, rectzoom, CV_RGB(255, 0, 0), 3, 8, 0);
        //imshow("ROI scaled", idepth_scaled_copy);
        idepth_scaled_copy=idepth_scaled.clone();

        //detect circle and isolate it
        Mat maskedZoom= CircleDetection(zoomeq);
        //namedWindow("Circled Hole", CV_WINDOW_AUTOSIZE);
        //imshow("Circled Hole", maskedZoom );


        // floodfill each isolated hole from the center
        Point seedpoint=cvPoint(roi_width/2, roi_height/2);
        Scalar inten=zoom.at<uchar>(roi_width/2, roi_height/2);
        //std::cout << "Greyscale center value:"<< inten.val[0] << endl;
        floodFill(maskedZoom,seedpoint,Scalar(255),0, 40,255, 4 | FLOODFILL_FIXED_RANGE);
        maskedZoom.convertTo(maskedZoom, CV_8UC1);
        //namedWindow("Zoom FloodFill", CV_WINDOW_AUTOSIZE);
        //imshow("Zoom FloodFill", maskedZoom);
        Point3d hole_center=cloudSubset(cloud,im_scan, maskedZoom,ROI_scan, ROI_msg,i); //point cloud crop
        //if (fp==true){holes.erase(holes.begin()+i);}

        if ((hole_center.x!=0) && (hole_center.y!=0) && (hole_center.z!=0)){
            centers_indcloud.push_back(hole_center);
        }

	}

       }
       else {
        //std::cout << "No holes found" << std::endl<<std::endl;

       }

  centersFile.close();
  cout << "Analizado agujero. Pulsar tecla para continuar" << endl;
  waitKey(0);
  //std::cout << "__________________________"<< endl;
  //std::cout << "Found holes:" <<centers_indcloud.size() << std::endl<<std::endl;
  //std::cout << "__________________________"<< endl;
  std::cout << "Found holes in cloud ["<< c <<"]=" << centers_indcloud.size() << std::endl;
  cout << "Centers ind. cloud" << endl;
  cout << centers_indcloud <<endl << endl;
  

  // Draw detected holes in ROI and complete image.
  Mat img_with_keypoints;
  drawKeypoints( idepth_scaled, holes, img_with_keypoints, Scalar(0,0,255),    DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  namedWindow("Holes Detected", CV_WINDOW_AUTOSIZE);
  //waitKey(0);

  for (int i=0; i<holes.size();i++){

      ostringstream convert;
      convert << i;
      string s = convert.str();
      //std::cout << "Points" << holes[i].pt <<std::endl;


      putText(img_with_keypoints, s, holes[i].pt, FONT_HERSHEY_PLAIN, 1.5, Scalar(0,0,255), 2);
  }
  imshow("Holes Detected", img_with_keypoints );


     //ros::Time end = ros::Time::now();
     //std::cout << "Time " << end-begin << endl;
     //cout << endl;
     //cout << "Hole detecting done" << endl;



  //Matrices de transformación
  Eigen::Matrix4f individual_tr = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f global_tr = Eigen::Matrix4f::Identity();

  std::stringstream ss_pcd;
  std::stringstream ss_depth;
  std::stringstream ss_binar;


  PointCloud::Ptr source (new PointCloud), target (new PointCloud);
  PointCloud::Ptr output (new PointCloud);



    /*if (i==clouds.size()){

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_pcd (new pcl::PointCloud<pcl::PointXYZRGB>);
      copyPointCloud (*show, *result_pcd);
      sensor_msgs::PointCloud2 cloud_tr;
      pcl::io::savePCDFile<pcl::PointXYZRGB>("/home/carba/Downloads/Archivos scanmatching mina/result.pcd", *result_pcd);
      pcl::toROSMsg(*show,cloud_tr);
      std_msgs::Header h = cloud_tr.header;
      cloud_tr.header.frame_id = "world";
      pub.publish (cloud_tr);
      break;
    }
    else{
      copyPointCloud (*clouds[i], *target);
      copyPointCloud (*clouds[i+1], *source);
    }*/
    if (c>0){
            cout << "ICP:" << endl;
            cout << endl;
            //cout << "dentro ICP " << endl;
            copyPointCloud (*clouds[c-1], *target);
            copyPointCloud (*clouds[c], *source);
            cout << "   Transform from cloud " << c << " to " << c-1 <<endl;

	    individual_tr = pairAlign (source, target);

	    GlobalTransform *= individual_tr; //Actualizamos GlobalTransform

	    pcl::transformPointCloud (*source, *output, GlobalTransform);

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centers (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr trnsfm_centers (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointXYZ centro_exis;
            std::vector<Point3d> new_centres;
            Point3d nuevo_centro;

            for (int h=0; h<centers_indcloud.size();h++){


                centro_exis.x=centers_indcloud[h].x;
                centro_exis.y=centers_indcloud[h].y;
                centro_exis.z=centers_indcloud[h].z;

                cloud_centers->push_back(centro_exis);

            }

            pcl::transformPointCloud (*cloud_centers, *trnsfm_centers, GlobalTransform);

            for (int h=0; h<trnsfm_centers->size();h++){
                pcl::PointXYZ centro_nuevo;
                bool remove=false;
                centro_nuevo.x=trnsfm_centers->points[h].x;
                centro_nuevo.y=trnsfm_centers->points[h].y;
                centro_nuevo.z=trnsfm_centers->points[h].z;

                cout << "centro transformado" << centro_nuevo << endl;

                for (int f=0; f<acc_centres.size();f++){

                    double x = centro_nuevo.x - acc_centres[f].x;
                    double y = centro_nuevo.y - acc_centres[f].y;
                    double z=  centro_nuevo.z - acc_centres[f].z;
                    double error_vector= sqrt(pow(x, 2) + pow(y, 2)+pow(z,2));
                    if (error_vector<0.15){
                        remove=true;
                        break;
                    }
                    nuevo_centro.x=centro_nuevo.x;
                    nuevo_centro.y=centro_nuevo.y;
                    nuevo_centro.z=centro_nuevo.z;

                }
                if (remove==false){
                    new_centres.push_back(nuevo_centro);
                    cout << "Nuevos centros añadidos" << nuevo_centro <<endl;
                }
                else{cout << "No hay centros nuevos" <<endl;}



            }

            for (int i=0; i<new_centres.size(); i++){
                acc_centres.push_back(new_centres[i]);

            }
            new_centres.clear();

            /*std::vector<Point3d> new_centres;
            Point3d nuevo_centro;

            for (int h=0; h<centers_indcloud.size();h++){
                bool remove=false;
                cout << "Centro previo" << centers_indcloud[h] <<endl;
                pcl::PointXYZ centro;
                std::vector<float> result;
                float calc;



                centro.x=centers_indcloud[h].x;
                centro.y=centers_indcloud[h].y;
                centro.z=centers_indcloud[h].z;

                cloud_centers->points.push_back(centro);

                //cout << "Nube orig" << *cloud_centers <<PointXYZendl;
                //pcl::transformPointCloud (*cloud_centers, *trnsfm_centers, GlobalTransform);

                for ( int i = 0; i < 4; ++i ){
                       calc = centro.x * GlobalTransform(0,i) + centro.y * GlobalTransform(1,i) + centro.z + GlobalTransform(2,i) + GlobalTransform(3,i);
                       result.push_back(calc);
                    }
                result[0] = result[0]/result[3];
                result[1] = result[1]/result[3];
                result[2] = result[2]/result[3];
                centro.x=result[0];
                centro.y=result[1];
                centro.z=result[2];

                //centro=trnsfm_centers->points[trnsfm_centers->points.size()];
                cout << "Centro transformado" << centro <<endl;
                //cout << "Nube transformad" << *trnsfm_centers <<endl;

                for (int f=0; f<acc_centres.size();f++){

                    double x = centro.x - acc_centres[f].x;
                    double y = centro.y - acc_centres[f].y;
                    double z=  centro.z - acc_centres[f].z;
                    double error_vector= sqrt(pow(x, 2) + pow(y, 2)+pow(z,2));
                    if (error_vector<0.05){
                        remove=true;
                        break;
                    }
                    nuevo_centro.x=centro.x;
                    nuevo_centro.y=centro.y;
                    nuevo_centro.z=centro.z;

                }
                if (remove==false){
                    new_centres.push_back(nuevo_centro);
                    cout << "Nuevos centros añadidos" << nuevo_centro <<endl;

                }
            }
            for (int i=0; i<new_centres.size(); i++){
                acc_centres.push_back(new_centres[i]);

            }
            new_centres.clear();*/




	    *show_prev += *output;

            pcl::VoxelGrid<PointT> grid;
	    grid.setLeafSize (0.02, 0.02, 0.02);
	    grid.setInputCloud (show_prev);
            grid.filter (*show_prev);

	    *show += *show_prev;

	    sensor_msgs::PointCloud2 cloud_tr;

	    pcl::toROSMsg(*show,cloud_tr);
	    std_msgs::Header h = cloud_tr.header;
            cloud_tr.header.frame_id = frame;
	    pub.publish (cloud_tr);

	    cout << "Nube final tiene " << show->size() << " puntos" << endl;


	    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_pcd (new pcl::PointCloud<pcl::PointXYZRGB>);
	    copyPointCloud (*show, *result_pcd);
            pcl::io::savePCDFile<pcl::PointXYZRGB>("result.pcd", *result_pcd);

	    cout << endl;
            cout << "    -Global Transform: " << endl;
            cout << GlobalTransform << endl << endl;
    }
   else if (c==0){acc_centres=centers_indcloud;}
   cout << "Global centers detected:" << endl;
   cout << acc_centres <<endl;

   /*sensor_msgs::PointCloud2 cloud_inicial;
   pcl::toROSMsg(*cloud,cloud_inicial);
   std_msgs::Header h = cloud_inicial.header;
   cloud_inicial.header.frame_id = frame;
   pub.publish (cloud_inicial);*/


  }


visualization_msgs::MarkerArray vec_array;

for (int j=0; j<acc_centres.size(); j++){
    geometry_msgs::Point center_msg;

        center_msg.x=acc_centres[j].x;
        center_msg.y=acc_centres[j].y;
        center_msg.z=acc_centres[j].z;
        pub_center.publish (center_msg);


        visualization_msgs::Marker loc_vector;
        geometry_msgs::Point initial_point,end_point;

        loc_vector.header.frame_id = frame;
        loc_vector.header.stamp = ros::Time();
        loc_vector.id = j;
        loc_vector.type = visualization_msgs::Marker::ARROW;
        loc_vector.action = visualization_msgs::Marker::ADD;


        initial_point.x = 0;
        initial_point.y = 0;
        initial_point.z = 0;
        //loc_vector.points[0].z = 0;

        loc_vector.points.push_back(initial_point);

        end_point.x = acc_centres[j].x;
        end_point.y = acc_centres[j].y;
        end_point.z = acc_centres[j].z;

        loc_vector.points.push_back(end_point);

        loc_vector.scale.x = 0.03;
        loc_vector.scale.y = 0.03;
        loc_vector.scale.z = 0.03;

        loc_vector.color.a = 1.0;
        loc_vector.color.r = 0.0;
        loc_vector.color.g = 1.0;
        loc_vector.color.b = 0.0;

        loc_vector.lifetime = ros::Duration();
        vec_array.markers.push_back(loc_vector);
        //pub_Arrow.publish(loc_vector);
        //waitKey(0);
}
    pub_Arrows.publish(vec_array);
    cout << "Fin " << endl;
    ros::shutdown();
  }



void cloud_cb (sensor_msgs::PointCloud2ConstPtr cloud_msg)// point cloud callback: kinect point cloud buffer
{
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
   pcl::fromROSMsg (*cloud_msg, *cloud);


   if (cloud->points.size () != 0){
    sensor_msgs::Image im_msg;
    cloudlist.push_back(cloud_msg);
    //cout << "dentro nube" << endl;
    for (int i=0;i<imglist.size();i++){
        if (imglist[i]->header.seq==cloud_msg->header.seq){
           im_msg=*imglist[i];

           clouds.push_back(cloud);
           //cout << "Nube prueba" << *cloud <<endl;
           Mat idepth=cv_bridge::toCvCopy(im_msg, "32FC1")->image; //conversion
           depthlist.push_back(idepth);

           cloudlist.clear();
           imglist.clear();

        }

    }
  }
  if (clouds.size()==nubes_a_analizar){
      //cout << "Nubes" << cloudlist.size() << endl;
      //cout << "Imagenes" << depthlist.size() << endl;
      processing();
  }
}




void depth_cb (const sensor_msgs::ImageConstPtr im_msg)//depth image callback: kinect depth image buffer
{
   if (im_msg->data.size () != 0){

   //cout << "dentro imagen" << endl;
   sensor_msgs::PointCloud2ConstPtr cloud_msg;
   imglist.push_back(im_msg);
   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
   for (int i=0;i<cloudlist.size();i++){
       if (cloudlist[i]->header.seq==im_msg->header.seq){
           cloud_msg=cloudlist[i];
           pcl::fromROSMsg (*cloud_msg, *cloud);
           clouds.push_back(cloud);
           //cout << "Nube prueba" << *cloud <<endl;
           Mat idepth=cv_bridge::toCvCopy(im_msg, "32FC1")->image; //conversion
           depthlist.push_back(idepth);


           cloudlist.clear();
           imglist.clear();

        }
   }
  }
   if (depthlist.size()==nubes_a_analizar){
       //cout << "Nubes" << cloudlist.size() << endl;
       //cout << "Imagenes" << depthlist.size() << endl;
       processing();
   }
}

int main (int argc, char** argv){



  ros::init (argc, argv, "detection");
  ros::NodeHandle n;
  centersFile.open("/home/jcl/Documents/Rosbags/centers.csv");

  // Create a ROS publisher for each hole ROI identifier
        pub = n.advertise<sensor_msgs::PointCloud2> ("cloud_tr",1); //Publish cloud transformed
        pub_ROIlocation = n.advertise<depth_deco::ROI_identifier> ("ROI_location", 1);
        // Create a ROS publisher for the ROI cloud
        pub_ROIcloud = n.advertise<sensor_msgs::PointCloud2> ("ROICloud", 1);
        // Create a ROS publisher for authentic center point of each hole
        pub_center = n.advertise<geometry_msgs::Point> ("HoleCenter", 1);
        // Create a ROS publisher for authentic Hole points
        pub_HoleCloud = n.advertise<sensor_msgs::PointCloud2> ("HoleCloud", 1);
        // Create a ROS publisher for the Rest cloud
        pub_Cloud = n.advertise<sensor_msgs::PointCloud2> ("Cloud", 1);
        // Create a ROS publisher for the Scan Matching cloud
        pub_ScanmCloud = n.advertise<sensor_msgs::PointCloud2> ("ScanmCloud", 1);
        // Create a ROS publisher for the Cylinder obtained through RANSAC
        pub_CylPoints = n.advertise<sensor_msgs::PointCloud2> ("CylinderPoints", 1);
        // Create a ROS publisher for DE-Optmimized Cylinder Model
        pub_Arrow = n.advertise<visualization_msgs::Marker> ("CenterVector", 1);
        pub_Arrows= n.advertise<visualization_msgs::MarkerArray>("HoleVectorArray", 1);


 //----------cargar desde topic-------


  int camera_sel;
  ros::Subscriber sub;
  ros::Subscriber sub2;

  while (camera_sel!=0 && camera_sel!=1){
  cout << "Select camera. Realsense = 0, Kinect = 1" << endl;
  cin >> camera_sel;
  }

  if (camera_sel==1){
      cout << "inside" << camera_sel << endl;
      sub = n.subscribe ("/kinect2/sd/image_depth", 1, depth_cb);
              // Create a ROS subscriber for the input point cloud
      sub2 = n.subscribe ("/kinect2/hd/points", 1, cloud_cb);
      frame="kinect2_ir_optical_frame";

  }
  else if (camera_sel==0){
      sub = n.subscribe ("/camera/depth/image_rect_raw", 1, depth_cb);
            // Create a ROS subscriber for the input point cloud
      sub2 = n.subscribe ("/camera/depth/color/points", 1, cloud_cb);
      frame="camera_depth_frame";
  }

  cout << "Nubes a analizar:" << endl;
  cin >> nubes_a_analizar;


//-------------cargar desde carpeta

      /*cout << endl;
      cout << "Cargando los archivos..." << endl;
      cout << "---------------------------" << endl;

      std::stringstream ss_pcd;
      std::stringstream ss_depth;
      std::stringstream ss_binar;

      for (int i = 1; i < nubes_a_analizar; i++){

    //pcd
        ss_pcd << "/home/carba/Documents/Archivos scanmatching mina/selpcd/" << i << ".pcd";
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZRGB>);
        //Load the point cloud
        pcl::io::loadPCDFile<pcl::PointXYZRGB>(ss_pcd.str(), *cloud_temp);
        //Once point cloud is loaded push back the point cloud into the vector
        clouds.push_back(cloud_temp);
        ss_pcd.str(std::string());

    //depth image
        ss_depth << "/home/carba/Documents/Archivos scanmatching mina/seldepthimg/" << i << ".jpg";
        Mat idepth = cv::imread (ss_depth.str(), 0);
        //Load the image
        depthlist.push_back(idepth);
        ss_depth.str(std::string());
        //cout  << " clouds loaded " << clouds.size() << endl;
    }
//Check sizes
  cout << clouds.size() << " clouds loaded " << endl;
  cout << depthlist.size() << " depth images loaded " << endl;
  processing();*/
//-----------------------------------------------






  ros::spin();

  return 0;
}



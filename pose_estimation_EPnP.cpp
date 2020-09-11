/*****************************************************************************
Author: Hongfei Li
Date:2020-02-18
Description: Estimate the transform matrix of world coordinates into tracker,
             and study the noise properties of the algorithm
******************************************************************************/
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace std;
using namespace cv;

/*
*Summary: project the gaze line(x,y,z) on tracker virtual plane
*Parameters:
*     gazeVec: A vector contains the gaze lines(x,y,z)
*Return : A vector contains projected gazes(x,y) which are 2D descriptors on virtual plane
*/
vector<Point2f> projectGazeToVirtualPlane(vector<Point3f> gazeVec)
{
    vector<Point2f> projectedGaze;
    Mat rVec = ( Mat_<float> ( 3,1 ) << 0, 0, 0);
    Mat tVec = ( Mat_<float> ( 3,1 ) << 0, 0, 0);
    Mat intrisicMat = ( Mat_<float> ( 3,3 ) << 1, 0, 0, 0, 1, 0, 0, 0, 1 );
    projectPoints(gazeVec, rVec, tVec, intrisicMat, Mat(), projectedGaze);
    return projectedGaze;
}

/*
*Summary: cross calibrate the transformation matrix of 3D world coordinates to eye tracker system
*Parameters:
*     points: 3D-points in world coordinates/stereoscopic vision system
*     projectedGaze: 2D image points on virtual plane
*     Mc: The matrix of camera intrinsic parameters
*Return : The matrix that transforms a 3D-point in world coordinates into the eye tracker system,
*         it is a 3 rows x 4 cols matrix, contains a 3D rotation and a 3D translation
*/
Mat crossCalibration(vector<Point3f> points, vector<Point2f> projectedGaze,Mat Mc)
{
    Mat r, t;
    solvePnP ( points, projectedGaze, Mc, Mat(), r, t, false, SOLVEPNP_EPNP ); // EPNP(1)
    Mat R;
    Rodrigues ( r, R );
    Mat Mt;
    hconcat(R, t, Mt);
    Mt.convertTo(Mt, CV_32FC1);
    return Mt;
}

/*
*Summary: convert the datas in Mat format to 3D-points' vector format
*Parameters:
*     row3Matrix: A 3-rows matrix
*Return : A vector contains 3D-points.
*/
vector<Point3f> convertMatToVec(Mat row3Matrix)
{
    vector<Point3f> retVec;
    for(int i=0;i<row3Matrix.cols;i++)
    {
        Point3f tempPoint={row3Matrix.at<float>(0, i),row3Matrix.at<float>(1, i),row3Matrix.at<float>(2, i)};
        retVec.push_back(tempPoint);
    }
    return retVec;
}

/*
*Summary: calculate the average squared norm difference of 2 group 3D-points
*Parameters:
*     trueValue: 3D-points detected by stereoscopic vision system
*     predValue: 3D-points calculated by reProjection Algorithm
*Return : the value of L2-norm difference
*/
float calcSquareError(vector<Point3f> trueValue, vector<Point3f> predValue)
{
    float fSum=0;
    for(int i=0;i<trueValue.size();i++)
    {
        float fTmpSquaErr=(trueValue[i].x -predValue[i].x)*(trueValue[i].x -predValue[i].x)
                + (trueValue[i].y -predValue[i].y)*(trueValue[i].y -predValue[i].y)
                + (trueValue[i].z -predValue[i].z)*(trueValue[i].z -predValue[i].z);
        fSum+=fTmpSquaErr;
    }
    return sqrt(fSum/trueValue.size());
}

/*
*Summary: Project the gaze lines to predict 3D-points positions in world coordinates
*Parameters:
*     gazeVec: A vector contains the gaze lines(x,y,z)
*     transformation : A transformation matrix of 3D world coordinates to eye tracker system
*Return : 3D-points in world coordinates
*/
vector<Point3f> reProjection(vector<Point3f> gazeVec, Mat transformation)
{
    float gazeVec3d[4][gazeVec.size()];
    for (size_t i = 0; i < gazeVec.size(); i++)
    {
        gazeVec3d[0][i]=gazeVec[i].x;
        gazeVec3d[1][i]=gazeVec[i].y;
        gazeVec3d[2][i]=gazeVec[i].z;
        gazeVec3d[3][i]=1;
    }
    Mat mGazeMat3d = Mat(4,gazeVec.size(),CV_32FC1,gazeVec3d);

    cv::Mat_<float> patch = (cv::Mat_<float>(1,4) << 0,0,0,1);
    vconcat(transformation,patch,transformation);

    Mat mReproj3dPoint;
    mReproj3dPoint.convertTo(mReproj3dPoint, CV_32FC1);
    mReproj3dPoint= transformation.inv() * mGazeMat3d;
    return convertMatToVec(mReproj3dPoint);
}

/*
*Summary: A Gaussian random number generator
*Parameters:
*     mean: The mean value of this Gaussian generator
*     sigma: Gaussian noise distribution value
*Return : the value after add Gaussian noise.
*/
double cal_normal_random(double mean, double sigma)
{
  double rand_num[12], r, r2, c1, c2, c3, c4, c5,gauss_rand ;
  unsigned long max ;
  int i ;
  max = 2147483647 ;
  for(i = 0 ; i < 12 ; i++)
  {
    rand_num[i] = ((double) random())/((double)(max));
  }
  c1 = 0.029899776 ;
  c2 = 0.008355968 ;
  c3 = 0.076542912 ;
  c4 = 0.252408784 ;
  c5 = 3.949846138 ;

  r = 0.0 ;
  for(i = 0 ; i < 12 ; i++)
    r += rand_num[i] ;
  r = (r-6.0)/4.0 ;
  r2 = r*r ;
  gauss_rand=((((c1*r2+c2)*r2+c3)*r2+c4)*r2+c5)*r ;
  return(mean+sigma*gauss_rand) ;
}

/*
*Summary: Add Gaussian noise on gaze lines
*Parameters:
*     gazeVec: A vector contains the gaze lines(x,y,z)
*     noiseRatio: The ratio of magnitude, for calculating the noise value
*Return : gaze lines with Gaussian noises
*/
vector<Point3f> addNoiseOnGaze(vector<Point3f> gazeVec, float noiseRatio)
{
    vector<Point3f> gazeVecNoise;
    for (size_t i = 0; i < gazeVec.size(); i++)
    {
        float m= sqrt(gazeVec[i].x * gazeVec[i].x + gazeVec[i].y * gazeVec[i].y + gazeVec[i].z * gazeVec[i].z);
        Point3f gazeNoise;
        gazeNoise.x= cal_normal_random(gazeVec[i].x, m * noiseRatio);
        gazeNoise.y= cal_normal_random(gazeVec[i].y, m * noiseRatio);
        gazeNoise.z= cal_normal_random(gazeVec[i].z, m * noiseRatio);
        gazeVecNoise.push_back(gazeNoise);
    }
    return gazeVecNoise;
}

/*
*Summary: get datas from data file, usually is txt file
*Parameters:
*     FileName: A string contains file path and file name
*     points: A vector collect 3D-points in world coordinates
*     gazes : A vector collect gaze lines(x,y,z)
*/
void readTxtFileIntoVector(string FileName, vector<Point3f>& points, vector<Point3f>& gazes)
{
    ifstream infile(FileName);
    string line;
    while (getline(infile, line))
    {
        vector<int> dataVec;
        char s[5];
        int sIndex=0;
        for(int i=0;i<line.length();i++)
        {
            char tempChar=line.at(i);
            if(tempChar=='-' || (tempChar>='0' && tempChar<='9'))
            {
                 s[sIndex++]=tempChar;
            }
            else
            {
                if(sIndex)
                {
                    dataVec.push_back(atoi(s));
                }
                memset(s,0,5*sizeof(char));
                sIndex=0;
            }
        }
        if(dataVec.size()==6)
        {
            Point3f point;
            point.x= dataVec[0];
            point.y= dataVec[1];
            point.z= dataVec[2];
            Point3f gaze;
            gaze.x= dataVec[3];
            gaze.y= dataVec[4];
            gaze.z= dataVec[5];
            points.push_back(point);
            gazes.push_back(gaze);
        }
        else
            cout<<"data not enough"<<endl;
    }
}

int main ( int argc, char** argv )
{
    /* get data from txt file */
    vector<Point3f> Pi;
    vector<Point3f> gi;
    vector<Point2f> gazeOnVirtualPlane;     
    string pointfile = "./data.txt";
    readTxtFileIntoVector(pointfile, Pi, gi);

    /* use cross-calibration approach to correctly estimate the matrix Mt */
    gazeOnVirtualPlane = projectGazeToVirtualPlane(gi);
    Mat Mc = ( Mat_<float> ( 3,3 ) << 1, 0, 0, 0, 1, 0, 0, 0, 1 );
    Mat Mt=crossCalibration(Pi,gazeOnVirtualPlane,Mc);

    /*
    *output the noiseless matrix Mt
    */
    cout<<"Mt:"<<endl<<Mt<<endl;

    /* set k from 1 to 19 */
    for(int k=1;k<20;k++)
    {
        /* for each measurement gi, add a zero mean Gaussian random noise and obtain noisy gi_bar */
        vector<Point3f> gi_bar= addNoiseOnGaze(gi,k*0.01);
        /* project gi_bar onto the virtual projective plane, to obtain noisy pi_bar */
        vector<Point2f> pi_bar= projectGazeToVirtualPlane(gi_bar);
        /* compute the noisy matrix Mt_bar with noisy points pi_bar and noiseless 3D points Pi */
        Mat Mt_bar=crossCalibration(Pi,pi_bar,Mc);
        /* obtain the 3D points into their noisy equivalents as Pi_bar= inverse(Mt_bar) * gi_bar */
        vector<Point3f> Pi_bar= reProjection(gi_bar, Mt_bar);
        /* calculates re-projection error */
        float reprojectionErrorWithNoise= calcSquareError(Pi, Pi_bar);

        /*
        *output the value of squared norm differences, for current noise levels - k
        */
        cout<<"k="<<k<<", reprojectionError="<<reprojectionErrorWithNoise<<endl;
    }
}


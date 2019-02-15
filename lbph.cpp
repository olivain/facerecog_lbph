/*
Facial recognition using LBPHFaceRecognizer / OpenCV3.3 / RPI 3B 

to run:
g++ -o lbph lbph.cpp `pkg-config opencv --cflags --libs` -D OPENCV_EXTRA_MODULES_PATH=/home/pi/opencv_contrib-3.3.0/modules -I/usr/local/include -lraspicam_cv -lmmal -lmmal_core -lmmal_util -lopencv_core -lopencv_highgui -w -fpermissive


*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <raspicam/raspicam_cv.h>

using namespace cv;
using namespace std;

#define SCALEFACTOR 1.1
#define MIN_NEIGHBORS 5
#define MIN_SIZE 30

#define MAX_SLIDER_VALUE 255
#define NUM_EIGEN_FACES 34


// Weights for the different eigenvectors
int sliderValues[NUM_EIGEN_FACES];

// Matrices for average (mean) and eigenvectors
Mat averageFace;
vector<Mat> eigenFaces;

// Read jpg files from the directory and create a csv
void readImages(string dirName, vector<Mat> &images) {
  cout << "Reading images from " << dirName;
  // Add slash to directory name if missing

  ifstream mfile ("data.csv");
  DIR *dir;
  struct dirent *ent;
  int count = 0;
  //image extensions
  string imgExt = "jpg";
  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL) {
    /* print all the files and directories within directory */
    // and save all as csv file with label ZERO (0)
    ofstream mfile;
    mfile.open("data.csv");

    while ((ent = readdir (dir)) != NULL) {
      if(strcmp(ent->d_name,".") == 0 || strcmp(ent->d_name,"..") == 0 ) {
        continue;
      }
      string fname = ent->d_name;
      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos) {
        string path = dirName + fname;
          // resize image
          /*Size size(180,180);//the dst image size
          resize(img,img,size);//resize image
          imwrite(path,img);*/
          // Convert images to floating point type
          cout << "image " << path << " : OK" << endl;
          mfile << path << ";0" << endl;

        }
      }
    }
    mfile.close();
    closedir(dir);

  }

void read_from_csv(vector<Mat>& images, vector<int>& labels) {
     string line, path, classlabel;
     std::ifstream file("data.csv", ifstream::in);
      while ( getline (file,line) ) {
          stringstream token(line);
          getline(token, path, ';');
          getline(token, classlabel);
          images.push_back(imread(path,0));
          labels.push_back(atoi(classlabel.c_str()));
      }
}

// Create data matrix from a vector of images
static  Mat createDataMatrix(const vector<Mat> &images) {
        cout << "Creating data matrix from images ...";
  // Allocate space for all images in one data matrix.
  // The size of the data matrix is
  //
  // ( w  * h  * 3, numImages )
  //
  // where,
  //
  // w = width of an image in the dataset.
  // h = height of an image in the dataset.
  // 3 is for the 3 color channels.
        Mat data(static_cast<int>(images.size()), images[0].rows * images[0].cols * 3, CV_32F);
  // Turn an image into one row vector in the data matrix
        for(unsigned int i = 0; i < images.size(); i++) {
          // Extract image as one long vector of size w x h x 3
          Mat image = images[i].reshape(1,1);
          // Copy the long vector into one row of the destm
          image.copyTo(data.row(i));
        }

    cout << " DONE" << endl;
  return data;
}


//////////////////////////////////

int main(int argc, char **argv) {

  raspicam::RaspiCam_Cv Camera;


  Camera.set ( cv::CAP_PROP_FRAME_WIDTH, 480 );
  Camera.set( cv::CAP_PROP_FRAME_HEIGHT, 320 );

if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
  // Directory containing images
  char fronthaar[] = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml";
  CascadeClassifier haar; // Set haar cascade datas to recognize a new face
  cout<<"loading haar cascade\n"<<endl;
  if(!haar.load(fronthaar)){
      cout<<"Can not load haar file !\n"<<endl;
  }
  string dirName = "/var/www/html/face_recognizer/data/";
  vector<Mat> images;
  vector<int> labels;
  std::vector<Rect> face_found;
  cv::Mat new_image;
Size size(180,180);
  //readImages(dirName, images);
  read_from_csv(images,labels);
  cout << "csv read !"<< endl;

  for(unsigned int i = 0; i < images.size(); i++) {
    images[i].reshape(1,1);
  }

  cout << "creating lbph face recognizer"<<endl;
  cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
  cout << "training lbph face recognizer"<<endl;
  model->train(images, labels);
  cout << "saving lbph face recognizer"<<endl;
  model->setThreshold(105.0);

  model->write("lbph.yml");
  cout << "yaml saved !"<< endl;

  // load yaml
  //model->read("lbph.yml");

  cv::Mat frame_gray;

  cout<<"reading test image\n"<<endl;
  while(1){
    Camera.grab();
    Camera.retrieve(new_image);
    

    cvtColor(new_image,frame_gray,COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    haar.detectMultiScale(frame_gray, face_found);
    if(!face_found.size()) {
        cout<<"can not find face in test picture !\n"<<endl;
    } else {
      rectangle(new_image, Point(face_found[0].x,face_found[0].y), Point(face_found[0].x+face_found[0].width,face_found[0].y+face_found[0].height), Scalar(255,0,255), 2,8,0);

      Mat ROI(frame_gray, Rect(face_found[0].x,face_found[0].y,face_found[0].width,face_found[0].height));
      Mat crop_face;
      ROI.copyTo(crop_face);
      resize(crop_face, crop_face, size);
      int prediction = -1;
      double confidence = 0.0;
      model->predict(crop_face, prediction, confidence);

      cout << "prediction : " << prediction << " / confidence : "<< confidence <<endl;
      cout << "Model Information:" << endl;
      string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
              model->getRadius(),
              model->getNeighbors(),
              model->getGridX(),
              model->getGridY(),
              model->getThreshold());
      cout << model_info << endl;
      // We could get the histograms for example:
      vector<Mat> histograms = model->getHistograms();
      // But should I really visualize it? Probably the length is interesting:
      cout << "Size of the histograms: " << histograms[0].total() << endl;

    }
    imshow("lbph",new_image);
    if(waitKey(30)>=0) { break; }
  }

}

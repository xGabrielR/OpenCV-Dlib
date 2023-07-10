#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <dlib/dnn.h>
#include <dlib/string.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace std;
using namespace dlib;

// ------- DLIB NET -----------
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<alevel0<alevel1<alevel2<alevel3<alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,input_rgb_image_sized<150>>>>>>>>>>>>>;


// TODO:
//   1. try elasticlient.
//   2. Insert data if lower than threshold to not duplicated vector on elastic.
//   3. Full Face Tracking.

int data_insert_to_elastic (string& string_vector) {                    // elastic url
    string input_embs = "curl -H \"Content-Type: application/json\" -XPOST 127.0.0.1:9200/img_emb/_doc -d ' {\"vector\": ["; 
    input_embs = input_embs + string_vector + "]}'";

    int r = system(input_embs.c_str());

    return r;
}

void get_embs (Mat& input_ims, Mat& faces, anet_type& net) {

    for (int i = 0; i < faces.rows; i++) {
        
        int x, y, w, h;

        x = int(faces.at<float>(i, 0));
        y = int(faces.at<float>(i, 1));
        w = int(faces.at<float>(i, 2));
        h = int(faces.at<float>(i, 3));

        // Out of bounds problem for trim bounds
        if(x >= 0 && y >= 0 && w + x < input_ims.cols && h + y < input_ims.rows) {
            Rect cutter(x, y, w, h);

            Mat img_crop;
            img_crop = input_ims(cutter);
            cv::resize(img_crop, img_crop, Size(150,150)); // Missing Align

            matrix<rgb_pixel> dlib_img;
            std::vector<matrix<rgb_pixel>> faces;
            dlib::assign_image(dlib_img, cv_image<rgb_pixel>(img_crop));
            faces.push_back(dlib_img);

            std::vector<matrix<float,0,1>> encoddings = net(faces);

            for (auto enc: encoddings) {
                // Cast matrix to vector
                std::vector<double> x(enc.begin(), enc.end());
                
                // Cast vector to string
                std::stringstream ss;
                for (auto value: x) {
                    ss << value << ",";
                }

                // remove last coma.
                std::string result = ss.str();
                result = result.substr(0, result.find_last_of(","));

                // 0 - Ok, 1 - Error.
                int insert_result = data_insert_to_elastic(result);
            }
        } 
    }
}

int main() {
    string path = "/home/grc/arep_cpp/open_cv_dlib/nets/";
    // Ptr<FaceDetectorYN> detector = FaceDetectorYN::create(path+"face_detection_yunet_2022mar.onnx", "", Size(320, 320), 0.9, 0.3, 5000);

    // github of yunet for definition.
    // Using "default" params
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
        path+"face_detection_yunet_2023mar.onnx", 
        "", cv::Size(320, 320), 
        0.6f, 0.3f, 
        5000, 0, 0
    );

    anet_type net;
    deserialize(path+"dlib_face_recognition_resnet_model_v1.dat") >> net;

    VideoCapture video(0);
    Mat frame;

    if (!video.isOpened()) {
        cout << "Erro Camera!" << endl;
        
        return -1;

    } else {

        while (true) {
            video >> frame;
            Mat m, grayscale, faces;

            cvtColor(frame, grayscale, CV_RGB2GRAY); 

            // Stacked frame for """RGB"""
            Mat channels[3] = {grayscale, grayscale, grayscale};
            cv::merge(channels, 3, m);

            resize(m, m, Size(320, 320));
            
            // Yunet detector
            detector->detect(m, faces);
            
            Mat result = m.clone();

            if (!faces.empty()) {
                get_embs(result, faces, net);
            }

            namedWindow("FRAME", WINDOW_NORMAL);
            resizeWindow("FRAME", 600,400);
            imshow("FRAME", result);

            waitKey(1);
        }
    }
    return 0;
}
